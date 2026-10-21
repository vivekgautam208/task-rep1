import copy, math
from .routing import forward_microbatch, backward_local_sums, add_window_aux_router_grad
from .optimizer import adamw, ema
from .ops import amp_microbatch_status, amp_window_status

def _public_mb(f):
    return {k:copy.deepcopy(v) for k,v in f.items() if not k.startswith("_")}

def _trainable_values(state, router_grad, expert_grads):
    values=list(router_grad)
    for block,grad in zip(state["expert_params"],expert_grads):
        if bool(block.get("trainable",True)):
            values.extend(grad)
    return values

class LegacySession:
    def __init__(self, workload):
        self.workload=copy.deepcopy(workload); self.cfg=self.workload["config"]
        self.state={k:copy.deepcopy(self.workload[k]) for k in ["router","expert_params","experts","scheduler","loss_scaler"]}
        self.windows={str(w["id"]):w for w in self.workload["windows"]}
        self.pending={}; self.archive={}; self.completed=[]; self.completed_by_window={}

    def forward(self,window_id,microbatch_id):
        wid=str(window_id); mid=str(microbatch_id); key=(wid,mid)
        if key in self.pending: return _public_mb(self.pending[key]["fwd"])
        if key in self.archive: return _public_mb(self.archive[key])
        if wid in self.completed_by_window: raise ValueError("unknown retry after completed window")
        mb=next(m for m in self.windows[wid]["microbatches"] if str(m["id"])==mid)
        f=forward_microbatch(self.state,wid,mb,self.cfg)
        self.pending[key]={"fwd":f,"backward":False}
        return _public_mb(f)

    def backward(self,window_id,microbatch_id):
        wid=str(window_id); mid=str(microbatch_id); key=(wid,mid)
        if wid in self.completed_by_window:
            if key in self.archive: return None
            raise ValueError("unknown retry after completed window")
        if key not in self.pending: raise ValueError("backward before forward")
        item=self.pending[key]
        if item["backward"]: return None
        rg,pg=backward_local_sums(self.state,item["fwd"],self.cfg)
        peak,overflow=amp_microbatch_status(_trainable_values(self.state,rg,pg),float(self.state["loss_scaler"]["scale"]))
        item.update(rg=rg,pg=pg,backward=True,local_scaled_peak=peak,local_overflow=overflow)
        return None

    def flush(self,window_id):
        wid=str(window_id)
        if wid in self.completed_by_window: return copy.deepcopy(self.completed_by_window[wid])
        w=self.windows[wid]; items=[]
        for mb in w["microbatches"]:
            key=(wid,str(mb["id"]))
            if key not in self.pending or not self.pending[key]["backward"]: raise ValueError("flush before all backward")
            items.append(self.pending[key])
        fwds=[i["fwd"] for i in items]; ww=math.fsum(float(f["token_weight"]) for f in fwds)
        rg=[0.0]*len(self.state["router"]["weight"]); pg=[[0.0]*len(p["weight"]) for p in self.state["expert_params"]]
        if ww>0.0:
            inv=1.0/ww
            for item in items:
                for i,v in enumerate(item["rg"]): rg[i]+=v*inv
                for p,g in enumerate(item["pg"]):
                    for i,v in enumerate(g): pg[p][i]+=v*inv
        aux=add_window_aux_router_grad(rg,fwds,ww,self.cfg)
        task=math.fsum((f["token_weight"]/ww)*f["task_loss"] for f in fwds) if ww else 0.0
        z=math.fsum((f["token_weight"]/ww)*f["z_loss"] for f in fwds) if ww else 0.0
        total=task+float(self.cfg["aux_coef"])*aux+float(self.cfg["z_coef"])*z
        sc=self.state["loss_scaler"]; sch=self.state["scheduler"]; sb=float(sc["scale"]); cb=int(sch["cursor"])
        final_values=_trainable_values(self.state,rg,pg)
        local_statuses=[(float(i.get("local_scaled_peak",0.0)),bool(i.get("local_overflow",False))) for i in items]
        peak,overflow=amp_window_status(local_statuses,final_values,sb)
        norm=math.hypot(*[float(v) for v in final_values]); clip=None if overflow else min(1.0,float(self.cfg["max_grad_norm"])/(norm+1e-30))
        lr=float(self.cfg["optimizer"]["base_lr"])*float(sch["multipliers"][cb])
        if overflow:
            sc["scale"]=max(float(sc["min_scale"]),sb*float(sc["backoff_factor"])); sc["growth_tracker"]=0; applied=False
        else:
            rg=[v*clip for v in rg]
            for p,block in enumerate(self.state["expert_params"]):
                if bool(block.get("trainable",True)): pg[p]=[v*clip for v in pg[p]]
            adamw(self.state["router"],rg,self.cfg["optimizer"],lr); ema(self.state["router"],self.cfg["ema_decay"])
            for block,g in zip(self.state["expert_params"],pg):
                if bool(block.get("trainable",True)):
                    adamw(block,g,self.cfg["optimizer"],lr); ema(block,self.cfg["ema_decay"])
            sch["cursor"]=(cb+1)%len(sch["multipliers"]); sc["growth_tracker"]=int(sc["growth_tracker"])+1
            if sc["growth_tracker"]>=int(sc["growth_interval"]):
                sc["scale"]=min(float(sc["max_scale"]),sb*float(sc["growth_factor"])); sc["growth_tracker"]=0
            applied=True
        rep={"window_id":w["id"],"active_tokens":sum(f["active_tokens"] for f in fwds),"token_weight":ww,"task_loss":task,"aux_loss":aux,"z_loss":z,"total_loss":total,"grad_norm":norm,"scaled_grad_peak":peak,"clip_coef":clip,"overflow":overflow,"update_applied":applied,"lr":lr,"scheduler_cursor_before":cb,"scheduler_cursor_after":int(sch["cursor"]),"loss_scale_before":sb,"loss_scale_after":float(sc["scale"]),"microbatches":[_public_mb(f) for f in fwds]}
        for mb in w["microbatches"]:
            key=(wid,str(mb["id"])); self.archive[key]=copy.deepcopy(self.pending[key]["fwd"]); self.pending.pop(key,None)
        self.completed.append(copy.deepcopy(rep)); self.completed_by_window[wid]=copy.deepcopy(rep)
        return copy.deepcopy(rep)

    def snapshot(self): return copy.deepcopy(self.state)

class Session:
    """Compatibility factory for scalar and residual-stack training sessions."""
    def __new__(cls, workload):
        if "model" in workload:
            from .stack_engine import Session as StackSession
            return StackSession(workload)
        return LegacySession(workload)

    @classmethod
    def restore(cls, workload, image):
        from .stack_engine import Session as StackSession
        return StackSession.restore(workload, image)


def run(workload):
    import json
    s = Session(workload)
    for w in workload["windows"]:
        for ev in w["events"]:
            op = ev["op"]
            if op == "forward":
                s.forward(w["id"], ev["microbatch_id"])
            elif op == "backward":
                s.backward(w["id"], ev["microbatch_id"])
            elif op == "ascend":
                s.ascend(w["id"])
            elif op == "prepare":
                s.prepare(w["id"])
            elif op == "flush":
                s.flush(w["id"])
            elif op == "restart":
                image = json.loads(json.dumps(s.checkpoint(), allow_nan=False))
                s = Session.restore(workload, image)
            else:
                raise ValueError("unknown event")
    return {"steps": s.completed, "final": s.snapshot()}

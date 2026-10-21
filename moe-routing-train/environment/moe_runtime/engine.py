import copy,math
from .routing import forward_microbatch,backward_sums
from .optimizer import grad_norm,adamw,ema,all_values
FP16_MAX=65504.0

def _pub(x):return {k:copy.deepcopy(v) for k,v in x.items() if not k.startswith("_")}
class LegacySession:
    def __init__(self,workload):
        self.workload=copy.deepcopy(workload);self.cfg=self.workload["config"];self.state={k:copy.deepcopy(self.workload[k]) for k in ["router","expert_params","experts","scheduler","loss_scaler"]};self.windows={str(w["id"]):w for w in self.workload["windows"]};self.pending={};self.completed=[]
    def forward(self,window_id,microbatch_id):
        w=self.windows[str(window_id)];mb=next(m for m in w["microbatches"] if str(m["id"])==str(microbatch_id));f=forward_microbatch(self.state,str(window_id),mb,self.cfg);self.pending[(str(window_id),str(microbatch_id))]={"fwd":f,"backward":False};return _pub(f)
    def backward(self,window_id,microbatch_id):
        key=(str(window_id),str(microbatch_id));item=self.pending[key];rg,pg=backward_sums(self.state,item["fwd"],self.cfg)
        tw=float(item["fwd"]["token_weight"]);den=tw if tw else 1.0;item["rg"]=[v/den for v in rg];item["pg"]=[[v/den for v in g] for g in pg];item["backward"]=True
    def flush(self,window_id):
        wid=str(window_id);w=self.windows[wid];items=[self.pending[(wid,str(m["id"]))] for m in w["microbatches"]];n=max(1,len(items));rg=[0.0]*len(self.state["router"]["weight"]);pg=[[0.0]*len(p["weight"]) for p in self.state["expert_params"]]
        for item in items:
            for i,v in enumerate(item["rg"]):rg[i]+=v/n
            for p,g in enumerate(item["pg"]):
                for i,v in enumerate(g):pg[p][i]+=v/n
        tw=sum(float(i["fwd"]["token_weight"]) for i in items); task=sum(i["fwd"]["task_loss"] for i in items)/n;aux=sum(i["fwd"].get("_aux_local",0.0) for i in items)/n;z=sum(i["fwd"]["z_loss"] for i in items)/n;total=task+float(self.cfg["aux_coef"])*aux+float(self.cfg["z_coef"])*z
        sc=self.state["loss_scaler"];sch=self.state["scheduler"];sb=float(sc["scale"]);cb=int(sch["cursor"]);lr=float(self.cfg["optimizer"]["base_lr"])*float(sch["multipliers"][cb]);sch["cursor"]=(cb+1)%len(sch["multipliers"])
        scaled=[abs(v*sb) for v in rg]
        peak=max(scaled,default=0.0);overflow=any((not math.isfinite(v)) or v>FP16_MAX for v in scaled);norm=grad_norm(rg,pg);clip=None if overflow else min(1.0,float(self.cfg["max_grad_norm"])/(norm+1e-30))
        if overflow:
            sc["scale"]=max(float(sc["min_scale"]),sb*float(sc["backoff_factor"]));sc["growth_tracker"]=0;applied=False
        else:
            for i in range(len(rg)):rg[i]*=clip
            for g in pg:
                local=math.sqrt(sum(v*v for v in g));c=min(1.0,float(self.cfg["max_grad_norm"])/(local+1e-30))
                for i in range(len(g)):g[i]*=c
            adamw(self.state["router"],rg,self.cfg["optimizer"],lr);ema(self.state["router"],self.cfg["ema_decay"])
            for e in self.state["experts"]:
                pid=int(e["param_id"]);adamw(self.state["expert_params"][pid],pg[pid],self.cfg["optimizer"],lr);ema(self.state["expert_params"][pid],self.cfg["ema_decay"])
            sc["growth_tracker"]+=1
            if sc["growth_tracker"]>=int(sc["growth_interval"]):sc["scale"]=min(float(sc["max_scale"]),sb*float(sc["growth_factor"]));sc["growth_tracker"]=0
            applied=True
        rep={"window_id":w["id"],"active_tokens":sum(i["fwd"]["active_tokens"] for i in items),"token_weight":tw,"task_loss":task,"aux_loss":aux,"z_loss":z,"total_loss":total,"grad_norm":norm,"scaled_grad_peak":peak,"clip_coef":clip,"overflow":overflow,"update_applied":applied,"lr":lr,"scheduler_cursor_before":cb,"scheduler_cursor_after":int(sch["cursor"]),"loss_scale_before":sb,"loss_scale_after":float(sc["scale"]),"microbatches":[_pub(i["fwd"]) for i in items]}
        for m in w["microbatches"]:self.pending.pop((wid,str(m["id"])),None)
        self.completed.append(copy.deepcopy(rep));return rep
    def snapshot(self):return copy.deepcopy(self.state)
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

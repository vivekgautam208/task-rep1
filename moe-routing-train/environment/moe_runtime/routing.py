import math
from .rng import CursorRng
from .ops import dot, softmax_with_lse, expert_forward, expert_backward

_CURSOR_BY_SEED={}
def _cursor(seed):
    return _CURSOR_BY_SEED.setdefault(int(seed),CursorRng(seed))

def _row(router,e,dim):
    s=e*(dim+1); return router["weight"][s:s+dim],router["weight"][s+dim]

def forward_microbatch(state, window_id, mb, cfg):
    active=[t for t in mb["tokens"] if t["mask"]]; E=len(state["experts"]); D=int(cfg["dim"]); H=int(cfg["hidden"]); k=int(cfg["top_k"])
    if not active:
        return {"microbatch_id":mb["id"],"active_tokens":0,"token_weight":0.0,"capacity":0,"accepted_counts":[0]*E,"dropped_counts":[0]*E,"task_loss":0.0,"z_loss":0.0,"routes":[],"_rows":[],"_aux_local":0.0}
    r=_cursor(cfg["rng_seed"]); token_weight=sum(float(t["weight"]) for t in active)
    rows=[]; queues={}
    for t in active:
        x=t["x"]; logits=[]
        for e in range(E):
            w,b=_row(state["router"],e,D); logits.append(dot(w,x)+b+float(cfg["router_jitter"])*(2*r.uniform()-1))
        probs,lse=softmax_with_lse(logits); selected=sorted(range(E),key=lambda e:logits[e],reverse=True)[:k]; mass=sum(probs[e] for e in selected); pre=[probs[e]/mass for e in selected]
        row={"token":t,"x":x,"probs":probs,"lse":lse,"selected":selected,"pre":pre,"accepted_slots":set(),"expert_y":{},"masks":{}}; ri=len(rows); rows.append(row)
        for slot,e in enumerate(selected): queues.setdefault((int(t["rank"]),e),[]).append((-pre[slot],int(t["ordinal"]),ri,slot))
    cap=max(1,math.ceil(float(cfg["capacity_factor"])*len(active)*k/E)); accepted=[0]*E; dropped=[0]*E
    for (rank,e),q in queues.items():
        q.sort()
        for pos,(_,_,ri,slot) in enumerate(q):
            if pos<cap: rows[ri]["accepted_slots"].add(slot); accepted[e]+=1
            else:dropped[e]+=1
    wp=[0.0]*E; wl=[0.0]*E; task_sum=z_sum=0.0; routes=[]
    keep_prob=1.0-float(cfg["expert_dropout"])
    for row in rows:
        t=row["token"]; wt=float(t["weight"])
        for e in range(E): wp[e]+=wt*row["probs"][e]
        acc=[e for slot,e in enumerate(row["selected"]) if slot in row["accepted_slots"]]
        for e in acc: wl[e]+=wt
        denom=sum(row["probs"][e] for e in acc) if acc else 1.0; gates=[row["probs"][e]/denom for e in acc]
        ys=[]
        for e in acc:
            pid=int(state["experts"][e]["param_id"]); mask=[r.uniform()>=float(cfg["expert_dropout"]) for _ in range(H)]
            y,_,_=expert_forward(state["expert_params"][pid]["weight"],row["x"],H,float(cfg["expert_dropout"]),mask); row["expert_y"][e]=y; row["masks"][e]=mask; ys.append(y)
        pred=sum(g*y for g,y in zip(gates,ys)); err=pred-float(t["target"]); row["accepted"]=acc;row["gates"]=gates;row["pred"]=pred;row["err"]=err
        task_sum+=wt*0.5*err*err;z_sum+=wt*row["lse"]*row["lse"]
        routes.append({"ordinal":int(t["ordinal"]),"uid":str(t["uid"]),"selected_experts":list(row["selected"]),"accepted_experts":acc,"accepted_gates":gates})
    pbar=[v/token_weight for v in wp];load=[v/(token_weight*k) for v in wl];aux=E*sum(pbar[e]*load[e] for e in range(E));task=task_sum/token_weight;z=z_sum/token_weight
    return {"microbatch_id":mb["id"],"active_tokens":len(active),"token_weight":token_weight,"capacity":cap,"accepted_counts":accepted,"dropped_counts":dropped,"task_loss":task,"z_loss":z,"routes":routes,"_rows":rows,"_load":load,"_aux_local":aux}

def backward_sums(state,fwd,cfg):
    E=len(state["experts"]);D=int(cfg["dim"]);H=int(cfg["hidden"]);rg=[0.0]*len(state["router"]["weight"]);pg=[[0.0]*len(p["weight"]) for p in state["expert_params"]]
    if not fwd["token_weight"]:return rg,pg
    for row in fwd["_rows"]:
        t=row["token"];wt=float(t["weight"]);p=row["probs"];dlog=[0.0]*E
        for e,g in zip(row["accepted"],row["gates"]):
            y=row["expert_y"][e];dlog[e]+=wt*row["err"]*g*(y-row["pred"]);pid=int(state["experts"][e]["param_id"])
            eg=expert_backward(state["expert_params"][pid]["weight"],row["x"],H,float(cfg["expert_dropout"]),row["masks"][e],wt*row["err"]*g)
            for i,v in enumerate(eg):pg[pid][i]+=v
        le=sum(p[e]*fwd["_load"][e] for e in range(E))
        for e in range(E):
            dlog[e]+=float(cfg["aux_coef"])*E*wt*p[e]*(fwd["_load"][e]-le)
            dlog[e]+=float(cfg["z_coef"])*2.0*wt*p[e]
            s=e*(D+1)
            for d in range(D):rg[s+d]+=dlog[e]*row["x"][d]
            rg[s+D]+=dlog[e]
    return rg,pg

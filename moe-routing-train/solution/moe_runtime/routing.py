import math
from .rng import router_jitter, expert_keep
from .ops import dot, softmax_with_lse, expert_forward, expert_backward


def _router_row(router, expert, dim):
    start = expert * (dim + 1)
    return router["weight"][start:start + dim], router["weight"][start + dim]


def _drop_mask(seed, window_id, microbatch_id, uid, logical_expert, hidden, dropout_p):
    return [expert_keep(seed, dropout_p, window_id, microbatch_id, uid, logical_expert, h)
            for h in range(hidden)]


def forward_microbatch(state, window_id, mb, cfg):
    active = sorted((t for t in mb["tokens"] if bool(t["mask"])), key=lambda t: int(t["ordinal"]))
    experts=state["experts"]; params=state["expert_params"]; router=state["router"]
    E=len(experts); D=int(cfg["dim"]); H=int(cfg["hidden"]); k=int(cfg["top_k"])
    seed=int(cfg["rng_seed"]); jitter=float(cfg["router_jitter"]); dropout=float(cfg["expert_dropout"])
    if not active:
        return {"microbatch_id":mb["id"],"active_tokens":0,"token_weight":0.0,"capacity":0,
                "accepted_counts":[0]*E,"dropped_counts":[0]*E,"task_loss":0.0,"z_loss":0.0,"routes":[],
                "_rows":[],"_weighted_prob":[0.0]*E,"_accepted_weight":[0.0]*E}
    token_weight=math.fsum(float(t["weight"]) for t in active)
    capacity=max(1,math.ceil(float(cfg["capacity_factor"])*len(active)*k/E))
    rows=[]; queues=[[] for _ in range(E)]
    for t in active:
        uid=str(t["uid"]); x=[float(v) for v in t["x"]]; logits=[]
        for e in range(E):
            wr,b=_router_row(router,e,D)
            logits.append(dot(wr,x)+b+router_jitter(seed,jitter,window_id,mb["id"],uid,e))
        probs,lse=softmax_with_lse(logits); selected=sorted(range(E),key=lambda e:logits[e],reverse=True)[:k]
        mass=math.fsum(probs[e] for e in selected); pre=[probs[e]/mass for e in selected]
        row={"token":t,"x":x,"probs":probs,"lse":lse,"selected":selected,"accepted_slots":set(),"expert_y":{},"masks":{}}
        ri=len(rows); rows.append(row)
        for slot,e in enumerate(selected): queues[e].append((-pre[slot],int(t["ordinal"]),uid,ri,slot))
    accepted=[0]*E; dropped=[0]*E
    for e,q in enumerate(queues):
        q.sort()
        for pos,(_,_,_,ri,slot) in enumerate(q):
            if pos<capacity: rows[ri]["accepted_slots"].add(slot); accepted[e]+=1
            else: dropped[e]+=1
    weighted_prob=[0.0]*E; accepted_weight=[0.0]*E; task_sum=z_sum=0.0; routes=[]
    for row in rows:
        t=row["token"]; wt=float(t["weight"])
        for e in range(E): weighted_prob[e]+=wt*row["probs"][e]
        acc=[]
        for slot,e in enumerate(row["selected"]):
            if slot in row["accepted_slots"]: acc.append(e); accepted_weight[e]+=wt
        denom=math.fsum(row["probs"][e] for e in acc) if acc else 1.0; gates=[row["probs"][e]/denom for e in acc]
        ys=[]
        for e in acc:
            pid=int(experts[e]["param_id"]); mask=_drop_mask(seed,window_id,mb["id"],str(t["uid"]),e,H,dropout)
            y,_,_=expert_forward(params[pid]["weight"],row["x"],H,dropout,mask); row["expert_y"][e]=y; row["masks"][e]=mask; ys.append(y)
        pred=math.fsum(g*y for g,y in zip(gates,ys)) if gates else 0.0; err=pred-float(t["target"])
        row.update(accepted=acc,gates=gates,pred=pred,err=err); task_sum+=wt*0.5*err*err; z_sum+=wt*row["lse"]*row["lse"]
        routes.append({"ordinal":int(t["ordinal"]),"uid":str(t["uid"]),"selected_experts":list(row["selected"]),"accepted_experts":acc,"accepted_gates":gates})
    return {"microbatch_id":mb["id"],"active_tokens":len(active),"token_weight":token_weight,"capacity":capacity,
            "accepted_counts":accepted,"dropped_counts":dropped,"task_loss":task_sum/token_weight,"z_loss":z_sum/token_weight,
            "routes":routes,"_rows":rows,"_weighted_prob":weighted_prob,"_accepted_weight":accepted_weight}


def backward_local_sums(state, fwd, cfg):
    E=len(state["experts"]); D=int(cfg["dim"]); H=int(cfg["hidden"]); dropout=float(cfg["expert_dropout"])
    rg=[0.0]*len(state["router"]["weight"]); pg=[[0.0]*len(p["weight"]) for p in state["expert_params"]]
    for row in fwd["_rows"]:
        t=row["token"]; wt=float(t["weight"]); dlog=[0.0]*E
        for e,g in zip(row["accepted"],row["gates"]):
            y=row["expert_y"][e]; dlog[e]+=wt*row["err"]*g*(y-row["pred"]); pid=int(state["experts"][e]["param_id"])
            eg=expert_backward(state["expert_params"][pid]["weight"],row["x"],H,dropout,row["masks"][e],wt*row["err"]*g)
            for i,v in enumerate(eg): pg[pid][i]+=v
        for e,p in enumerate(row["probs"]): dlog[e]+=float(cfg["z_coef"])*2.0*wt*row["lse"]*p
        for e in range(E):
            s=e*(D+1)
            for d in range(D): rg[s+d]+=dlog[e]*row["x"][d]
            rg[s+D]+=dlog[e]
    return rg,pg


def add_window_aux_router_grad(router_grad, forwards, window_weight, cfg):
    if window_weight<=0.0: return 0.0
    E=len(forwards[0]["_weighted_prob"]) if forwards else 0; k=int(cfg["top_k"]); D=int(cfg["dim"])
    prob=[0.0]*E; accepted=[0.0]*E
    for f in forwards:
        for e in range(E): prob[e]+=f["_weighted_prob"][e]; accepted[e]+=f["_accepted_weight"][e]
    pbar=[v/window_weight for v in prob]; load=[v/(window_weight*k) for v in accepted]
    aux=E*math.fsum(pbar[e]*load[e] for e in range(E))
    coef=float(cfg["aux_coef"])*E/window_weight
    for f in forwards:
        for row in f["_rows"]:
            wt=float(row["token"]["weight"]); probs=row["probs"]; expect=math.fsum(probs[e]*load[e] for e in range(E))
            for e in range(E):
                dl=coef*wt*probs[e]*(load[e]-expect); s=e*(D+1)
                for d in range(D): router_grad[s+d]+=dl*row["x"][d]
                router_grad[s+D]+=dl
    return aux

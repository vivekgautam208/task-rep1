import copy, hashlib, math
FP16_MAX=65504.0

def u01(seed,*parts):
    key="|".join([str(int(seed))]+[str(p) for p in parts]).encode();raw=hashlib.blake2b(key,digest_size=8,person=b"moe-rng-v1").digest();return int.from_bytes(raw,"big")/18446744073709551616.0
def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))
def sm(x):
    m=max(x);e=[math.exp(v-m) for v in x];d=math.fsum(e);return [v/d for v in e],m+math.log(d)
def ef(w,x,H,p,mask):
    D=len(x);a0=H*D;b0=a0+H;q=1/(1-p) if p<1 else 0;zs=[];hd=[]
    for h in range(H):
        a=math.fsum(w[h*D+d]*x[d] for d in range(D))+w[a0+h];z=math.tanh(a);zs.append(z);hd.append(z*q if mask[h] else 0.0)
    return math.fsum(w[b0+h]*hd[h] for h in range(H)),zs,hd
def eb(w,x,H,p,mask,up):
    D=len(x);a0=H*D;b0=a0+H;q=1/(1-p) if p<1 else 0;y,zs,hd=ef(w,x,H,p,mask);g=[0.0]*len(w)
    for h in range(H):
        g[b0+h]+=up*hd[h]
        if mask[h]:
            da=up*w[b0+h]*q*(1-zs[h]*zs[h]);g[a0+h]+=da
            for d in range(D):g[h*D+d]+=da*x[d]
    return g
def mbeval(state,wid,mb,cfg):
    A=sorted((t for t in mb["tokens"] if t["mask"]),key=lambda t:int(t["ordinal"]));E=len(state["experts"]);D=int(cfg["dim"]);H=int(cfg["hidden"]);k=int(cfg["top_k"]);seed=int(cfg["rng_seed"]);jit=float(cfg["router_jitter"]);dp=float(cfg["expert_dropout"])
    if not A:return {"microbatch_id":mb["id"],"active_tokens":0,"token_weight":0.0,"capacity":0,"accepted_counts":[0]*E,"dropped_counts":[0]*E,"task_loss":0.0,"z_loss":0.0,"routes":[]},[0.0]*len(state["router"]["weight"]),[[0.0]*len(p["weight"]) for p in state["expert_params"]],[0.0]*E,[0.0]*E,[]
    W=math.fsum(float(t["weight"]) for t in A);cap=max(1,math.ceil(float(cfg["capacity_factor"])*len(A)*k/E));rows=[];qs=[[] for _ in range(E)]
    for t in A:
        x=[float(v) for v in t["x"]];log=[]
        for e in range(E):
            s=e*(D+1);noise=jit*(2*u01(seed,"router_jitter",str(wid),str(mb["id"]),str(t["uid"]),int(e))-1);log.append(dot(state["router"]["weight"][s:s+D],x)+state["router"]["weight"][s+D]+noise)
        p,lse=sm(log);sel=sorted(range(E),key=lambda e:log[e],reverse=True)[:k];mass=math.fsum(p[e] for e in sel);pre=[p[e]/mass for e in sel];r={"t":t,"x":x,"p":p,"lse":lse,"sel":sel,"slots":set()};ri=len(rows);rows.append(r)
        for sl,e in enumerate(sel):qs[e].append((-pre[sl],int(t["ordinal"]),str(t["uid"]),ri,sl))
    ac=[0]*E;dr=[0]*E
    for e,q in enumerate(qs):
        q.sort()
        for pos,(_,_,_,ri,sl) in enumerate(q):
            if pos<cap:rows[ri]["slots"].add(sl);ac[e]+=1
            else:dr[e]+=1
    wp=[0.0]*E;aw=[0.0]*E;task=zsum=0.0;routes=[];rg=[0.0]*len(state["router"]["weight"]);pg=[[0.0]*len(p["weight"]) for p in state["expert_params"]]
    for r in rows:
        t=r["t"];wt=float(t["weight"]);p=r["p"]
        for e in range(E):wp[e]+=wt*p[e]
        acc=[e for sl,e in enumerate(r["sel"]) if sl in r["slots"]]
        for e in acc:aw[e]+=wt
        den=math.fsum(p[e] for e in acc) if acc else 1.0;gates=[p[e]/den for e in acc];ys=[];masks=[]
        for e in acc:
            mask=[u01(seed,"expert_dropout",str(wid),str(mb["id"]),str(t["uid"]),int(e),int(h))>=dp for h in range(H)];pid=int(state["experts"][e]["param_id"]);y,_,_=ef(state["expert_params"][pid]["weight"],r["x"],H,dp,mask);ys.append(y);masks.append(mask)
        pred=math.fsum(g*y for g,y in zip(gates,ys)) if gates else 0.0;err=pred-float(t["target"]);task+=wt*.5*err*err;zsum+=wt*r["lse"]**2;r.update(acc=acc,gates=gates,ys=ys,masks=masks,pred=pred,err=err)
        routes.append({"ordinal":int(t["ordinal"]),"uid":str(t["uid"]),"selected_experts":list(r["sel"]),"accepted_experts":acc,"accepted_gates":gates})
        dl=[0.0]*E
        for e,g,y,mask in zip(acc,gates,ys,masks):
            dl[e]+=wt*err*g*(y-pred);pid=int(state["experts"][e]["param_id"]);eg=eb(state["expert_params"][pid]["weight"],r["x"],H,dp,mask,wt*err*g)
            for i,v in enumerate(eg):pg[pid][i]+=v
        for e in range(E):
            dl[e]+=float(cfg["z_coef"])*2*wt*r["lse"]*p[e];s=e*(D+1)
            for d in range(D):rg[s+d]+=dl[e]*r["x"][d]
            rg[s+D]+=dl[e]
    return {"microbatch_id":mb["id"],"active_tokens":len(A),"token_weight":W,"capacity":cap,"accepted_counts":ac,"dropped_counts":dr,"task_loss":task/W,"z_loss":zsum/W,"routes":routes},rg,pg,wp,aw,rows
def adam(b,g,opt,lr):
    b["step"]+=1;t=b["step"];b1=float(opt["beta1"]);b2=float(opt["beta2"]);c1=1-b1**t;c2=1-b2**t;eps=float(opt["eps"]);rate=lr*float(b.get("lr_scale",1));wd=float(opt["weight_decay"])*float(b.get("decay_scale",1))
    for i,x in enumerate(g):
        old=b["weight"][i];m=b1*b["m"][i]+(1-b1)*x;v=b2*b["v"][i]+(1-b2)*x*x;b["m"][i]=m;b["v"][i]=v;b["weight"][i]=old*(1-rate*wd)-rate*(m/c1)/(math.sqrt(v/c2)+eps)
def ema(b,d):
    for i,w in enumerate(b["weight"]):b["ema"][i]=d*b["ema"][i]+(1-d)*w
def reference_run(workload):
    if "model" in workload:
        from stack_reference import reference_stack
        return reference_stack(workload)
    cfg=workload["config"];state={k:copy.deepcopy(workload[k]) for k in ["router","expert_params","experts","scheduler","loss_scaler"]};steps=[];E=len(state["experts"]);D=int(cfg["dim"]);k=int(cfg["top_k"])
    for w in workload["windows"]:
        vals=[mbeval(state,str(w["id"]),mb,cfg) for mb in w["microbatches"]];reps=[x[0] for x in vals];W=math.fsum(r["token_weight"] for r in reps);rg=[0.0]*len(state["router"]["weight"]);pg=[[0.0]*len(p["weight"]) for p in state["expert_params"]];wp=[0.0]*E;aw=[0.0]*E
        if W:
            for rep,rgi,pgi,wpi,awi,rows in vals:
                for i,v in enumerate(rgi):rg[i]+=v/W
                for p,g in enumerate(pgi):
                    for i,v in enumerate(g):pg[p][i]+=v/W
                for e in range(E):wp[e]+=wpi[e];aw[e]+=awi[e]
            pbar=[v/W for v in wp];load=[v/(W*k) for v in aw];aux=E*math.fsum(pbar[e]*load[e] for e in range(E));coef=float(cfg["aux_coef"])*E/W
            for *_,rows in vals:
                for r in rows:
                    wt=float(r["t"]["weight"]);p=r["p"];ex=math.fsum(p[e]*load[e] for e in range(E))
                    for e in range(E):
                        dl=coef*wt*p[e]*(load[e]-ex);s=e*(D+1)
                        for d in range(D):rg[s+d]+=dl*r["x"][d]
                        rg[s+D]+=dl
            task=math.fsum((r["token_weight"]/W)*r["task_loss"] for r in reps);z=math.fsum((r["token_weight"]/W)*r["z_loss"] for r in reps)
        else:aux=task=z=0.0
        total=task+float(cfg["aux_coef"])*aux+float(cfg["z_coef"])*z;sc=state["loss_scaler"];sch=state["scheduler"];sb=float(sc["scale"]);cb=int(sch["cursor"]);allv=list(rg)
        for b,g in zip(state["expert_params"],pg):
            if bool(b.get("trainable",True)):allv.extend(g)
        local_peak=0.0;local_over=False
        for rep,rgi,pgi,*_ in vals:
            lv=list(rgi)
            for b,g in zip(state["expert_params"],pgi):
                if bool(b.get("trainable",True)):lv.extend(g)
            for x in lv:
                q=abs(x*sb)
                if not math.isfinite(q):local_over=True;local_peak=float("inf")
                else:local_peak=max(local_peak,q)
                if q>FP16_MAX:local_over=True
        scaled=[abs(x*sb) for x in allv];final_peak=max(scaled,default=0);peak=max(local_peak,final_peak);overflow=local_over or any((not math.isfinite(x)) or x>FP16_MAX for x in scaled);norm=math.hypot(*allv);clip=None if overflow else min(1.0,float(cfg["max_grad_norm"])/(norm+1e-30));lr=float(cfg["optimizer"]["base_lr"])*float(sch["multipliers"][cb])
        if overflow:sc["scale"]=max(float(sc["min_scale"]),sb*float(sc["backoff_factor"]));sc["growth_tracker"]=0;ap=False
        else:
            rg=[x*clip for x in rg];adam(state["router"],rg,cfg["optimizer"],lr);ema(state["router"],float(cfg["ema_decay"]))
            for b,g in zip(state["expert_params"],pg):
                if bool(b.get("trainable",True)):adam(b,[x*clip for x in g],cfg["optimizer"],lr);ema(b,float(cfg["ema_decay"]))
            sch["cursor"]=(cb+1)%len(sch["multipliers"]);sc["growth_tracker"]+=1
            if sc["growth_tracker"]>=int(sc["growth_interval"]):sc["scale"]=min(float(sc["max_scale"]),sb*float(sc["growth_factor"]));sc["growth_tracker"]=0
            ap=True
        steps.append({"window_id":w["id"],"active_tokens":sum(r["active_tokens"] for r in reps),"token_weight":W,"task_loss":task,"aux_loss":aux,"z_loss":z,"total_loss":total,"grad_norm":norm,"scaled_grad_peak":peak,"clip_coef":clip,"overflow":overflow,"update_applied":ap,"lr":lr,"scheduler_cursor_before":cb,"scheduler_cursor_after":int(sch["cursor"]),"loss_scale_before":sb,"loss_scale_after":float(sc["scale"]),"microbatches":reps})
    return {"steps":steps,"final":state}

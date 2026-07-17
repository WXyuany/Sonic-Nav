#!/usr/bin/env python3
"""Train the custom hybrid PPO policy on the fast residual curriculum."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch

SCRIPTS = Path(__file__).resolve().parent.parent
if str(SCRIPTS) not in sys.path: sys.path.insert(0, str(SCRIPTS))
from sonic_world.rl import HybridPPO, HybridRecurrentActorCritic, RolloutBatch
from sonic_world.rl.hybrid_ppo import generalized_advantage
from sonic_world.rl.residual_env import WorldModelResidualEnv

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--iterations',type=int,default=50); p.add_argument('--num-envs',type=int,default=256); p.add_argument('--horizon',type=int,default=64); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--output',default='reports/policy_models/world_model_hybrid_ppo_v0.pt'); a=p.parse_args()
    env=WorldModelResidualEnv(a.num_envs,device=a.device); policy=HybridRecurrentActorCritic().to(a.device); algo=HybridPPO(policy)
    for it in range(a.iterations):
        entity,context=env.observe(); rows=[]
        for _ in range(a.horizon):
            e,c=entity.unsqueeze(1),context.unsqueeze(1); act,rec,lp,val,_=policy.act(e,c); (entity,context),rew,done,_=env.step(act[:,0].detach(),rec[:,0].detach()); rows.append((e,c,act.detach(),rec.detach(),lp.detach(),val.detach(),rew,done))
        ent=torch.cat([x[0] for x in rows],1); ctx=torch.cat([x[1] for x in rows],1); act=torch.cat([x[2] for x in rows],1); rec=torch.cat([x[3] for x in rows],1); lp=torch.cat([x[4] for x in rows],1); val=torch.cat([x[5] for x in rows],1); rew=torch.stack([x[6] for x in rows],1); done=torch.stack([x[7] for x in rows],1); adv,ret=generalized_advantage(rew,val,done); metrics=algo.update(RolloutBatch(ent,ctx,act,rec,lp,adv,ret,torch.ones_like(done,dtype=torch.bool)))
        if it % 10 == 0: print(f"iter={it} reward={rew.mean().item():.3f} loss={metrics['loss']:.3f}")
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); torch.save({'schema':'sonic_world_model_hybrid_ppo_v0','state_dict':policy.state_dict(),'observation':'entity12x2+context24','continuous_actions':8,'recovery_actions':5},out); print(out); return 0
if __name__=='__main__': raise SystemExit(main())

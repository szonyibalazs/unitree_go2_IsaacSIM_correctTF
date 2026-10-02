"""Go2 lépcső-policy tanítás a stairs4 világban, height scan (elevation pontfelhő) obs-szal.
  python train_stairs.py --num_envs 16 --max_iterations 1500 --headless
Alapból a régi rough checkpointból indul (finetune); --scratch: nulláról."""
import argparse, glob, os, sys, types
from datetime import datetime
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--max_iterations", type=int, default=1500)
p.add_argument("--init", type=str, default=None, help="régi (model_state_dict) checkpoint; alap: legújabb unitree_go2_rough")
p.add_argument("--scratch", action="store_true")
p.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.terrain = "stairs4"
sys.modules["omniverse_sim"] = types.SimpleNamespace(args_cli=args)  # custom_rl_env ezt importálja
sim_app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
from custom_rl_env import UnitreeGo2Stairs4TrainEnvCfg

env_cfg = UnitreeGo2Stairs4TrainEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env_cfg.scene.height_scanner.debug_vis = not args.headless  # sárga pontok = amit a policy lát
env_cfg.commands.base_velocity.debug_vis = not args.headless  # nyíl a robot felett = a haladási parancs
env = RslRlVecEnvWrapper(ManagerBasedRLEnv(env_cfg), clip_actions=5.0)  # scale 0.25 -> max ±1.25 rad az alapálltól; enélkül a policy elszáll

cfg = UnitreeGo2RoughPPORunnerCfg()
cfg.experiment_name, cfg.seed, cfg.max_iterations = "unitree_go2_stairs4", args.seed, args.max_iterations
cfg = handle_deprecated_rsl_rl_cfg(cfg, "5.0.1")
log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=log_dir, device=cfg.device)

if not args.scratch:
    init = args.init or sorted(glob.glob("logs/rsl_rl/unitree_go2_rough/*/model_*.pt"))[-1]
    print(f"[INFO] finetune innen: {init}")
    old = torch.load(init, map_location=cfg.device, weights_only=False)["model_state_dict"]
    for net, name in ((runner.alg.actor, "actor"), (runner.alg.critic, "critic")):
        sd = {"mlp." + k[len(name) + 1:]: v for k, v in old.items() if k.startswith(name + ".")}
        if name == "actor":
            sd["distribution.std_param"] = old["std"].clamp(min=0.2)  # NOTE: ha a std kulcsa/neve eltér, a load hibázik -> nézd meg a print-et
        print(name, net.load_state_dict(sd, strict=False))

runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
env.close()
sim_app.close()

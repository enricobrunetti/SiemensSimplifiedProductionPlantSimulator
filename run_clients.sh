
n_agents=3
base_port=9900
base_seed=1000
scp=ai_optimizer/configs/simulator_config_"$n_agents"units.json
pcp=ai_optimizer/configs/simulator_config_"$n_agents"units.json
lcp=ai_optimizer/configs/learning_config.json
for ((i=0;i<$n_agents;i+=1))
  do
      python3 ai_optimizer/cppu_path_learner_decentralized.py --addr localhost --port $((base_port + i)) --scp $scp --product_config_path $pcp --lcp $lcp --workers 0 --cppu cppu_$i --algo SAC --checkpoint 0 --outdir output/ --seed $((base_seed + i)) &
done
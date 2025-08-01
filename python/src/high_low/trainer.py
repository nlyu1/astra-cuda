class HighLowVTraceTrainer:
    def __init__(self, args, agent, name='', checkpoint_interval=10_000_000, device='cuda'):
        self.args = args 
        self.agent = agent 
        self.optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=args.learning_rate,
            eps=1e-5,
        )
        self.last_checkpoint = -1
        self.checkpoint_interval = checkpoint_interval
        self.name = name 

    def train(self, update_dictionary, global_step):
        obs, logprobs, actions, rewards, dones = (
            update_dictionary['obs'],
            update_dictionary['logprobs'],
            update_dictionary['actions'].long(),
            update_dictionary['rewards'],
            update_dictionary['dones'],
        )

        # Unlike in PPO, we batch sample by environment instead 
        batch_env_indices = np.arange(self.args.num_envs)
        explained_vars, value_losses, pg_losses, entropy_losses, approx_kls = [], [], [], [], []
        (T, B), single_obs_shape = obs.shape[:2], obs.shape[2:]
        # Obs: [T, B, s]. Logprobs: [T, B], Actions: [T, B, 4], Rewards [T, B], Done: [T, B]

        for _ in range(self.args.update_epochs):
            np.random.shuffle(batch_env_indices)
            for start in range(0, self.args.num_envs, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                minibatch_env_indices = batch_env_indices[start:end]

                # Off-policy network, using surrogate loss 
                # [T, B, 4] -> [T*B, 4]
                reference_actions = actions[:, minibatch_env_indices].reshape(-1, 4)
                batch_obs = obs[:, minibatch_env_indices].reshape(-1, *single_obs_shape)
                reference_actions = {
                    'bid_price': reference_actions[:, 0],
                    'ask_price': reference_actions[:, 1],
                    'bid_size': reference_actions[:, 2],
                    'ask_size': reference_actions[:, 3],
                }
                _, new_logprob, entropy, values = self.agent(
                    batch_obs, reference_actions)
                # Recast back to normal shape 
                new_logprob = new_logprob.reshape(T, B)
                entropy = entropy.reshape(T, B)
                values = values.reshape(T, B)

                with torch.no_grad():
                    log_ratio = new_logprob - logprobs[:, minibatch_env_indices]
                    ratio = log_ratio.exp() 
                    approx_kl = ((ratio - 1) - log_ratio).mean()


                assert dones[-1, minibatch_env_indices].all(), "All episodes must be terminated at the end of the episode"
                augmented_values = torch.cat([ # [T+1, B]
                    values, torch.zeros_like(values[-1])[None, :]])                 
                
                vtrace_results = compiled_vtrace_losses(
                    rewards[:, minibatch_env_indices],
                    dones[:, minibatch_env_indices],
                    logprobs[:, minibatch_env_indices],
                    new_logprob,
                    augmented_values,
                    gamma=self.args.gamma)

                entropy_loss = -entropy.mean()
                loss = (vtrace_results['policy_loss'] 
                        + self.args.ent_coef * entropy_loss 
                        + vtrace_results['value_loss'] * self.args.vf_coef)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

                explained_vars.append(vtrace_results['value_r2'].item())
                value_losses.append(vtrace_results['value_loss'].item())
                pg_losses.append(vtrace_results['policy_loss'].item())
                entropy_losses.append(-entropy.mean().item())
                approx_kls.append(approx_kl.item())

        wandb.log({
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/value_loss": np.mean(value_losses),
            "losses/policy_loss": np.mean(pg_losses),
            "losses/entropy": np.mean(entropy_losses),
            "losses/approx_kl": np.mean(approx_kls),
            "losses/value_returns_r2": np.nanmean(explained_vars),
        }, step=global_step)

    def save_checkpoint(self, step):
        if step < self.last_checkpoint + self.checkpoint_interval:
            return 
        self.last_checkpoint = step

        checkpoint_path = f"checkpoints/{self.args.exp_name}_{step}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        print('Saving checkpoint to', checkpoint_path)
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'args': self.args,
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
            }
        }, checkpoint_path)
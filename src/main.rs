use rand::prelude::*;
use rand_distr::StandardNormal;
use std::collections::HashMap;
use std::fmt;

fn main() 
{
    let bd1 = Bandit::new(0.0,7.0);
    let bd2 = Bandit::new(2.0,3.0);
    let bd3 = Bandit::new(4.0,2.0);
    let bd4 = Bandit::new(5.0,0.0);
    let bd5 = Bandit::new(4.0,13.0);

    let bandits : [Bandit; 5] = [bd1, bd2, bd3, bd4, bd5];
    let agent: Agent = Agent::new(&bandits);
    agent.step()
}

struct Agent
{
    score: f64,
    action_qualities: HashMap<Bandit, f64>,
    actions: u32,
}

impl Agent
{
    fn new(bandits: &[Bandit]) -> Self { 
        let mut action_qualities = HashMap::new();
        for b in bandits
        {
            action_qualities.insert(*b, 0.0);
        }
        return Self{
            score: 0.0,
            action_qualities: action_qualities,
            actions: 0,
        }
    }

    fn step(mut self)
    {
        let chosen_bandit = self.pick_action();
        let reward = &chosen_bandit.reward();
        self.score += reward;
        self.update_action_qualities(chosen_bandit, reward)
    }


    fn update_action_qualities(&mut self, chosen_bandit: Bandit, reward: &f64)
    {
        let old_val = self.action_qualities[&chosen_bandit];
        let new_val = old_val + (1/self.actions) as f64*(reward-old_val);
        self.action_qualities.insert(chosen_bandit,new_val);
    }

    fn pick_action(&self) -> Bandit
    {
        let mut best_quality = 0.0;
        let mut best_bandit: Bandit = Bandit { mean_bits: 0, std_bits: 0 };
        for b in self.action_qualities.keys()
        {
            if &self.action_qualities[b] > &best_quality
            {
                best_quality = self.action_qualities[b];
                best_bandit = *b;
            }   
        }
        best_bandit
    }

}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct Bandit
{
    // stored as u64 to make hashable
    mean_bits: u64, 
    std_bits: u64
}

// allows printing to the terminal
impl fmt::Display for Bandit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(mean: {}, std: {})", self.mean(), self.std())
    }
}


impl Bandit
{
    fn new(mean: f64, std: f64) -> Self
    {
        return Self{mean_bits: mean.to_bits(), std_bits: std.to_bits()};
    }

    fn mean(&self) -> f64
    {
        f64::from_bits(self.mean_bits)
    }

    fn std(&self) -> f64
    {
        f64::from_bits(self.std_bits)
    }

    fn reward(&self) -> f64
    {
        let mut reward = thread_rng().sample::<f64,_>(StandardNormal);
        reward = reward + self.mean() * self.std();
        reward
    }
}
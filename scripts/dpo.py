# The_RLHF_via_Direct_Preference_Optimization

def dpo_loss(beta, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):
    """
    Direct Preference Optimization Loss.
    Encourages the model to maximize the gap between 'good' and 'bad' answers 
    relative to a static reference model.
    """
    # Calculate the log-ratio between the current policy and the frozen reference model
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    logits = pi_logratios - ref_logratios
    
    # Loss = -log(sigmoid(beta * logits))
    loss = -F.logsigmoid(beta * logits).mean()
    return loss

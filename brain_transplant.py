
import torch
import copy
import os

def brain_transplant(model_path, output_path, donor_id=4, recipient_ids=[0, 3]):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    print(f"Loading corrupted model from {model_path}...")
    # Load with weights_only=False because it contains numpy/objects
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    agents_state_dict = checkpoint['agents_state_dict']
    
    print(f"Donor ID: {donor_id}")
    print(f"Recipient IDs: {recipient_ids}")
    
    # Get donor brain
    donor_brain = agents_state_dict[donor_id]
    
    # Perform surgery
    for recipient_id in recipient_ids:
        print(f"Transplanting brain from Drone {donor_id} to Drone {recipient_id}...")
        # Deep copy to ensure independence later (though for inference it doesn't matter much)
        agents_state_dict[recipient_id] = copy.deepcopy(donor_brain)
        
    # Save new model
    checkpoint['agents_state_dict'] = agents_state_dict
    torch.save(checkpoint, output_path)
    print(f"Operation successful! Saved fixed model to {output_path}")

if __name__ == "__main__":
    # Define paths
    original_model = "swarm_training_results/models/model_episode_1600.pth"
    fixed_model = "swarm_training_results/models/model_episode_1600_fixed.pth"
    
    # Drone 4 was observed to be very effective (destroyed Target 8 and others)
    # Drone 0 and 3 were stuck in corners
    brain_transplant(original_model, fixed_model, donor_id=4, recipient_ids=[0, 3])

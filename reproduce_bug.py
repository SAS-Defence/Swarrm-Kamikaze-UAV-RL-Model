
import unittest
from Environment import SwarmBattlefield2D
from SwarmCoordinator import SwarmCoordinator

class TestSwarmLogic(unittest.TestCase):
    def setUp(self):
        self.env = SwarmBattlefield2D(num_drones=5, num_targets=5)
        # Force specific config to make testing deterministic
        self.env.target_types['tank']['required_drones'] = 2
        self.coordinator = SwarmCoordinator(self.env)
        self.env.reset()
        self.coordinator.reset()

    def test_dead_drone_assignment(self):
        """Test if dead drones are removed from assignments"""
        # Step 1: Assign drones
        obs = self.env.get_observations()
        directives = self.coordinator.get_strategic_actions(obs)
        
        # Check assignments
        assigned_count = 0
        assigned_target = None
        for t_id, d_ids in self.coordinator.target_assignments.items():
            if len(d_ids) > 0:
                assigned_count += len(d_ids)
                assigned_target = t_id
                print(f"Target {t_id} assigned to drones: {d_ids}")
                break
        
        self.assertGreater(assigned_count, 0, "Drones should be assigned")
        
        # Step 2: Kill one assigned drone (simulate battery death)
        victim_drone_id = self.coordinator.target_assignments[assigned_target][0]
        print(f"Killing drone {victim_drone_id}")
        self.env.drones[victim_drone_id]['destroyed'] = True
        self.env.drones[victim_drone_id]['battery'] = 0
        
        # Step 3: Run coordinator update
        obs = self.env.get_observations()
        self.coordinator.get_strategic_actions(obs)
        
        # Step 4: Check if dead drone is still assigned
        current_assigns = self.coordinator.target_assignments[assigned_target]
        print(f"Assignments after death: {current_assigns}")
        
        self.assertNotIn(victim_drone_id, current_assigns, "Dead drone should be removed from assignments")

    def test_premature_idle(self):
        """Test if drones go idle while target is still alive"""
        # Step 1: Assign
        obs = self.env.get_observations()
        self.coordinator.get_strategic_actions(obs)
        
        # Find an active assignment
        target_id = None
        drone_id = None
        for t, d_ids in self.coordinator.target_assignments.items():
            if d_ids:
                target_id = t
                drone_id = d_ids[0]
                break
                
        if target_id is None:
            self.skipTest("No assignments made")
            
        print(f"Tracking Drone {drone_id} on Target {target_id}")
        
        # Step 2: Simulate steps ensuring target is NOT destroyed
        for _ in range(5):
            # Verify target is alive
            target = next(t for t in self.env.targets if t['id'] == target_id)
            self.assertFalse(target['destroyed'], "Target should be alive")
            
            # Update coordinator
            obs = self.env.get_observations()
            self.coordinator.get_strategic_actions(obs)
            
            # Check drone state in coordinator
            state = self.coordinator.drone_states[drone_id]
            self.assertEqual(state['status'], 'assigned', f"Drone {drone_id} went {state['status']} prematurely!")
            self.assertEqual(state['target'], target_id)
            
        print("Drone stayed assigned correctly.")

if __name__ == '__main__':
    unittest.main()

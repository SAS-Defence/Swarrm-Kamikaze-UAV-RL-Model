import numpy as np
import math
import random
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt


class SwarmBattlefield2D:
    def __init__(self, width=1000, height=800, num_drones=10, num_targets=15):
        self.width = width
        self.height = height
        self.num_drones = num_drones
        self.num_targets = num_targets

        # Sensor ve iletişim menzilleri
        self.sensor_range = 200  # piksel
        self.communication_range = 300
        self.attack_range = 15  # Kamikaze saldırı menzili (yakın mesafe!)

        # Drone hareket hızı
        self.drone_speed = 8.0

        # Hedef tipleri ve özellikleri - KAMİKAZE DRONE için optimize
        self.target_types = {
            'tank': {'hp': 3, 'importance': 15, 'color': 'darkgreen', 'radius': 25, 'required_drones': 3},
            'artillery': {'hp': 2, 'importance': 12, 'color': 'brown', 'radius': 20, 'required_drones': 2},
            'infantry': {'hp': 1, 'importance': 5, 'color': 'lightblue', 'radius': 10, 'required_drones': 1},
            'aircraft': {'hp': 3, 'importance': 14, 'color': 'gray', 'radius': 22, 'required_drones': 3},
            'radar': {'hp': 2, 'importance': 10, 'color': 'orange', 'radius': 15, 'required_drones': 2}
        }

        # Hedef dağılımı (gerçekçi)
        self.target_distribution = {
            'tank': 0.25,  # %25
            'artillery': 0.20,  # %20
            'infantry': 0.30,  # %30
            'aircraft': 0.15,  # %15
            'radar': 0.10  # %10
        }

        self.reward_params = {
            # YOK ETME ÖDÜLLERİ - ARTIRILDI
            'destroy_reward_multiplier': 100,  # Tank yok etmek = 100 * 15 = 1500 puan!
            
            # KULLANICI İSTEĞİ: "Sadece imha değil, hasar da ödül getirsin"
            # STRATEJİ: Ölüm Cezası (-15) < Hasar Ödülü (35)
            # Böylece drone tanka çarpıp ölse bile karlı çıkar (+20 Net).
            'damage_reward': 50, 

            # YAKLAŞMA VE KEŞİF
            'proximity_reward': 0.1,
            'discovery_reward': 1,

            # CEZALAR
            'movement_penalty': -0.001,
            'battery_penalty': -0.05,
            'death_penalty': -15,  # Hasar verip ölmek karlı olmalı

            # TAKIM ÇALIŞMASI
            'teamwork_bonus': 15,
            'efficiency_bonus': 20,

            # HAYATTA KALMA
            'survival_bonus': 0.01
        }

        # Ortam değişkenleri
        self.targets = []
        self.drones = []
        self.time_step = 0
        self.max_steps = 500

        # Metrikler
        self.metrics = {
            'total_destroyed': 0,
            'total_damage': 0,
            'total_discoveries': 0,
            'total_movement': 0
        }
        
        self.engaged_target_ids = set()

        print(f"[ENV] Harita: {width}x{height}, {num_drones} drone, {num_targets} hedef")

    def reset(self):
        """Ortamı sıfırla"""
        self.targets = []
        self.drones = []
        self.time_step = 0
        self.metrics = {k: 0 for k in self.metrics.keys()}
        self.engaged_target_ids.clear()

        self._deploy_targets_strategically()
        self._deploy_drones()

        initial_observations = self.get_observations()
        print(f"[ENV] Reset: {len(self.targets)} hedef ({self._get_target_summary()})")
        return initial_observations

    def _deploy_targets_strategically(self):
        """Hedefleri stratejik olarak yerleştir"""
        target_count = 0
        for target_type, probability in self.target_distribution.items():
            type_count = max(1, int(self.num_targets * probability))
            for _ in range(type_count):
                if target_count >= self.num_targets: break
                
                if target_type == 'tank':
                    x = random.randint(self.width * 2 // 3, self.width - 100)
                    y = random.randint(100, self.height - 100)
                elif target_type == 'artillery':
                    x = random.randint(self.width * 3 // 4, self.width - 50)
                    y = random.randint(150, self.height - 150)
                elif target_type == 'radar':
                    x = random.randint(self.width // 2, self.width - 200)
                    y = random.randint(50, self.height // 3)
                elif target_type == 'aircraft':
                    x = random.randint(self.width // 2, self.width - 150)
                    y = random.randint(self.height // 2, self.height - 100)
                else:
                    x = random.randint(self.width // 3, self.width - 100)
                    y = random.randint(100, self.height - 100)

                self.targets.append({
                    'id': target_count, 'type': target_type, 'x': x, 'y': y,
                    'hp': self.target_types[target_type]['hp'],
                    'max_hp': self.target_types[target_type]['hp'],
                    'importance': self.target_types[target_type]['importance'],
                    'required_drones': self.target_types[target_type]['required_drones'],
                    'destroyed': False, 'detected': False, 'detected_by': set(),
                    'attackers': set(), 'damage_taken': 0
                })
                target_count += 1
        
        while len(self.targets) < self.num_targets:
            target_type = random.choice(list(self.target_types.keys()))
            x = random.randint(self.width // 2, self.width - 100)
            y = random.randint(100, self.height - 100)
            self.targets.append({
                'id': len(self.targets), 'type': target_type, 'x': x, 'y': y,
                'hp': self.target_types[target_type]['hp'],
                'max_hp': self.target_types[target_type]['hp'],
                'importance': self.target_types[target_type]['importance'],
                'required_drones': self.target_types[target_type]['required_drones'],
                'destroyed': False, 'detected': False, 'detected_by': set(),
                'attackers': set(), 'damage_taken': 0
            })

    def _deploy_drones(self):
        """Drone'ları başlangıç pozisyonuna yerleştir"""
        deployment_zones = [
            (50, 100, 200, 300),
            (50, 100, self.height // 2 - 100, self.height // 2 + 100),
            (50, 100, self.height - 300, self.height - 100)
        ]
        for i in range(self.num_drones):
            zone_idx = i % len(deployment_zones)
            min_x, max_x, min_y, max_y = deployment_zones[zone_idx]
            self.drones.append({
                'id': i, 'x': random.randint(min_x, max_x), 'y': random.randint(min_y, max_y),
                'status': 'free', 'target_id': None, 'team': i % 3,
                'destroyed': False, 'detected_targets': set(), 'shared_targets': set(),
                'battery': 100.0, 'health': 100.0, 'total_damage': 0, 'total_kills': 0,
                'total_movement': 0.0, 'last_action': None
            })

    def _get_target_summary(self):
        summary = {}
        for target in self.targets:
            ttype = target['type']
            summary[ttype] = summary.get(ttype, 0) + 1
        return ', '.join([f'{k}:{v}' for k, v in summary.items()])

    def get_state(self):
        return {
            'targets': [t.copy() for t in self.targets],
            'drones': [d.copy() for d in self.drones],
            'time_step': self.time_step,
            'metrics': self.metrics.copy()
        }

    def get_observations(self):
        observations = []
        for drone in self.drones:
            if drone['destroyed']:
                observations.append(self._get_dead_drone_observation(drone['id']))
                continue

            obs = {
                'drone_id': drone['id'],
                'position': [drone['x'] / self.width, drone['y'] / self.height],
                'status': self._encode_status(drone['status']),
                'battery': drone['battery'] / 100.0,
                'health': drone['health'] / 100.0,
                'team': drone['team'] / 3.0,
                'visible_targets': [], 'visible_target_count': 0,
                'shared_targets': [], 'shared_target_count': 0,
                'teammates': [], 'teammate_count': 0,
                'assigned_target': -1, 'target_distance': 1.0, 'target_importance': 0.0
            }

            visible_targets = []
            for target in self.targets:
                if target['destroyed']: continue
                dist = self._calculate_distance(drone['x'], drone['y'], target['x'], target['y'])
                if dist <= self.sensor_range:
                    visible_targets.append({
                        'id': target['id'],
                        'type': self._encode_target_type(target['type']),
                        'distance': dist / self.sensor_range,
                        'direction_x': (target['x'] - drone['x']) / self.sensor_range,
                        'direction_y': (target['y'] - drone['y']) / self.sensor_range,
                        'importance': target['importance'] / 15.0,
                        'hp': target['hp'] / target['max_hp'],
                        'required_drones': target['required_drones'] / 3.0,
                        'attackers': len(target['attackers']) / target['required_drones']
                    })
            obs['visible_targets'] = visible_targets
            obs['visible_target_count'] = len(visible_targets) / 10.0

            shared_targets = set()
            teammates = []
            for other_drone in self.drones:
                if other_drone['id'] == drone['id'] or other_drone['destroyed']: continue
                dist = self._calculate_distance(drone['x'], drone['y'], other_drone['x'], other_drone['y'])
                if dist <= self.communication_range:
                    shared_targets.update(other_drone['detected_targets'])
                    if other_drone['team'] == drone['team']:
                        teammates.append({
                            'id': other_drone['id'],
                            'distance': dist / self.communication_range,
                            'direction_x': (other_drone['x'] - drone['x']) / self.communication_range,
                            'direction_y': (other_drone['y'] - drone['y']) / self.communication_range,
                            'status': self._encode_status(other_drone['status']),
                            'battery': other_drone['battery'] / 100.0,
                            'health': other_drone['health'] / 100.0,
                            'target_id': other_drone.get('target_id', -1)
                        })
            
            obs['shared_targets'] = list(shared_targets)[:10]
            obs['shared_target_count'] = len(shared_targets) / 10.0
            obs['teammates'] = teammates
            obs['teammate_count'] = len(teammates) / (self.num_drones // 3)

            if drone['target_id'] is not None:
                target = next((t for t in self.targets if t['id'] == drone['target_id']), None)
                if target and not target['destroyed']:
                    obs['assigned_target'] = target['id']
                    dist = self._calculate_distance(drone['x'], drone['y'], target['x'], target['y'])
                    obs['target_distance'] = dist / self.height # 800px üzerinden normalize (daha güvenli)
                    
                    # GPS EKLENTİSİ: Hedef yönü (Görünmese bile)
                    if dist > 0:
                        obs['target_direction_x'] = (target['x'] - drone['x']) / dist
                        obs['target_direction_y'] = (target['y'] - drone['y']) / dist
                    else:
                        obs['target_direction_x'] = 0.0
                        obs['target_direction_y'] = 0.0

                    # NORMAL ÖNEM SKORU (Blood in the Water İPTAL EDİLDİ)
                    obs['target_importance'] = target['importance'] / 15.0

            observations.append(obs)
        return observations

    def _get_dead_drone_observation(self, drone_id):
        return {
            'drone_id': drone_id, 'position': [0, 0], 'status': 0, 'battery': 0, 'health': 0,
            'team': 0, 'visible_targets': [], 'visible_target_count': 0,
            'shared_targets': [], 'shared_target_count': 0, 'teammates': [], 'teammate_count': 0,
            'assigned_target': -1, 'target_distance': 1.0, 'target_importance': 0.0
        }

    def _encode_status(self, status):
        return {'free': 0.25, 'engaged': 0.5, 'attacking': 0.75, 'destroyed': 0.0}.get(status, 0.0)

    def _encode_target_type(self, target_type):
        return {'infantry': 0.2, 'radar': 0.4, 'artillery': 0.6, 'aircraft': 0.8, 'tank': 1.0}.get(target_type, 0.0)

    def _calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def step(self, actions):
        rewards = [0.0] * len(self.drones)
        # 1. Hareket
        self._move_drones(actions, rewards)
        # 2. Keşif
        self._update_detection(rewards)
        # 3. Saldırı
        self._process_attacks(actions, rewards)
        # 4. Durum
        self._update_drone_status()
        # 5. Ek Ödüller
        self._calculate_additional_rewards(rewards)
        # 6. Bitiş
        done = self._check_done()
        # 7. Info
        global_info = self._get_global_info()
        self.time_step += 1
        return self.get_observations(), rewards, done, global_info

    def _move_drones(self, actions, rewards):
        for i, drone in enumerate(self.drones):
            if drone['destroyed']: continue
            if i < len(actions):
                action = actions[i]
                if len(action) >= 2:
                    move_x, move_y = max(-1.0, min(1.0, float(action[0]))), max(-1.0, min(1.0, float(action[1])))
                    delta_x, delta_y = move_x * self.drone_speed, move_y * self.drone_speed
                    movement = math.sqrt(delta_x ** 2 + delta_y ** 2)
                    drone['total_movement'] += movement
                    self.metrics['total_movement'] += movement
                    drone['x'] = max(0, min(drone['x'] + delta_x, self.width))
                    drone['y'] = max(0, min(drone['y'] + delta_y, self.height))
                    rewards[i] += self.reward_params['movement_penalty'] * movement
                    drone['battery'] = max(0.0, drone['battery'] - 0.05)
            if drone['battery'] <= 0:
                drone['destroyed'] = True; drone['status'] = 'destroyed'
                rewards[i] += self.reward_params['death_penalty']

    def _update_detection(self, rewards):
        for drone in self.drones:
            if drone['destroyed']: continue
            prev_detected = drone['detected_targets'].copy()
            drone['detected_targets'].clear()
            for target in self.targets:
                if target['destroyed']: continue
                if self._calculate_distance(drone['x'], drone['y'], target['x'], target['y']) <= self.sensor_range:
                    drone['detected_targets'].add(target['id'])
                    if drone['id'] not in target['detected_by']:
                        target['detected_by'].add(drone['id']); target['detected'] = True
                        rewards[drone['id']] += self.reward_params['discovery_reward']
                        self.metrics['total_discoveries'] += 1
            if drone['detected_targets'] - prev_detected:
                rewards[drone['id']] += len(drone['detected_targets'] - prev_detected) * self.reward_params['discovery_reward'] * 0.5

    def _process_attacks(self, actions, rewards):
        for i, drone in enumerate(self.drones):
            if drone['destroyed'] or i >= len(actions): continue
            # PROXIMITY FUZE (Otomatik Patlatma)
            # Eğer menzildeyse ve drone yaşıyorsa patlat! Trigger beklemeye gerek yok.
            target_id = -1
            
            # En yakın hedefi bul (Hitbox kontrolü)
            # Not: Zaten detected_targets vs var ama kesin çarpışma için mesafe bakıyoruz.
            for t in self.targets:
                if t['destroyed']: continue
                dist = self._calculate_distance(drone['x'], drone['y'], t['x'], t['y'])
                hitbox = t.get('radius', 15) + 10 # Tolerans artırıldı (+10)
                
                if dist <= hitbox:
                    target_id = t['id']
                    # STANDART HASAR (Reverted to Hard Mode)
                    damage = 1.0 
                    t['hp'] -= damage
                    t['damage_taken'] += damage
                    target = t # Link
                    drone['total_damage'] += damage
                    self.metrics['total_damage'] += damage
                    target['attackers'].add(drone['id'])
                    self.engaged_target_ids.add(target_id)
                    drone['status'] = 'attacking'
                    drone['target_id'] = target_id

                    reward_val = self.reward_params['damage_reward']
                    rewards[i] += reward_val
                    
                    drone['destroyed'] = True; drone['status'] = 'destroyed'; drone['health'] = 0
                    
                    is_kill = target['hp'] <= 0
                    if is_kill:
                         kill_reward = target['importance'] * self.reward_params['destroy_reward_multiplier']
                         print(f"[ENV] 💥 Drone {drone['id']} hedefe çarptı ve İMHA ETTİ! Ödül: {kill_reward}")
                    else:
                         print(f"[ENV] 💥 Drone {drone['id']} hedefe çarptı (Hasar)! Ödül: {reward_val}")

                    if is_kill:
                        target['destroyed'] = True; target['hp'] = 0
                        drone['total_kills'] += 1
                        self.metrics['total_destroyed'] += 1
                        destroy_reward = target['importance'] * self.reward_params['destroy_reward_multiplier']
                        for attacker_id in target['attackers']:
                            if attacker_id < len(rewards):
                                rewards[attacker_id] += destroy_reward if attacker_id == drone['id'] else destroy_reward * 0.3
                        if len(target['attackers']) >= target['required_drones']:
                            for attacker_id in target['attackers']:
                                if attacker_id < len(rewards):
                                    rewards[attacker_id] += self.reward_params['teamwork_bonus'] * len(target['attackers'])
                        print(f"[ENV] ✅ Hedef {target_id} YOK EDİLDİ!")
                    
                    # Çarpışma olduysa döngüden çık (Bir drone aynı anda tek hedefe çarpar)
                    break 
            
            # Eğer çarpışma olmadıysa proximity reward ver
            if target_id == -1:
                 # action[3] hala target_id olabilir mi? 
                 # Artık action trigger yok, sadece mesafe var.
                 pass

    def _update_drone_status(self):
        for target in self.targets:
            if target['destroyed']: target['attackers'].clear()
        for drone in self.drones:
            if drone['destroyed']: continue
            if drone['target_id'] is not None:
                target = next((t for t in self.targets if t['id'] == drone['target_id']), None)
                if target is None or target['destroyed']: drone['status'] = 'free'; drone['target_id'] = None

    def _calculate_additional_rewards(self, rewards):
        for i, drone in enumerate(self.drones):
            if drone['destroyed']: continue
            rewards[i] += self.reward_params['survival_bonus']
            if drone['battery'] < 20: rewards[i] += self.reward_params['battery_penalty']

    def _check_done(self):
        return all(t['destroyed'] for t in self.targets) or all(d['destroyed'] for d in self.drones) or self.time_step >= self.max_steps

    def _get_global_info(self):
        engaged_count = len(self.engaged_target_ids)
        destroyed_engaged = sum(1 for t_id in self.engaged_target_ids if self.targets[t_id]['destroyed'])
        return {
            'time_step': self.time_step, 'destroyed_targets': sum(1 for t in self.targets if t['destroyed']),
            'success_rate': (destroyed_engaged / engaged_count * 100) if engaged_count > 0 else 0,
            **self.metrics
        }

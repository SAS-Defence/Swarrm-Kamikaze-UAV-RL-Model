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
        self.attack_range = 40  # saldırı menzili

        # Drone hareket hızı
        self.drone_speed = 8.0

        # Hedef tipleri ve özellikleri (ZORLU)
        self.target_types = {
            'tank': {'hp': 4, 'importance': 15, 'color': 'darkgreen', 'radius': 25, 'required_drones': 3},
            'artillery': {'hp': 3, 'importance': 12, 'color': 'brown', 'radius': 20, 'required_drones': 2},
            'infantry': {'hp': 1, 'importance': 5, 'color': 'lightblue', 'radius': 10, 'required_drones': 1},
            'aircraft': {'hp': 3, 'importance': 14, 'color': 'gray', 'radius': 22, 'required_drones': 2},
            'radar': {'hp': 2, 'importance': 10, 'color': 'orange', 'radius': 15, 'required_drones': 1}
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
            'destroy_reward_multiplier': 30,  # Tank yok etmek = 30 * 15 = 450 puan!
            'damage_reward': 8,  # Her hasar = 8 puan

            # YAKLAŞMA VE KEŞİF - ARTIRILDI
            'proximity_reward': 1.0,  # Hedefe yaklaşma ödülü
            'discovery_reward': 5,  # Yeni hedef keşfi = 5 puan

            # CEZALAR - AZALTILDI (öğrenmeyi kolaylaştırır)
            'movement_penalty': -0.001,  # Hareket cezası çok düşük
            'battery_penalty': -0.05,  # Batarya cezası yumuşatıldı
            'death_penalty': -15,  # Ölüm cezası azaltıldı

            # TAKIM ÇALIŞMASI - ARTIRILDI
            'teamwork_bonus': 15,  # Koordineli saldırı = 15 puan
            'efficiency_bonus': 20,  # Verimlilik bonusu artırıldı

            # HAYATTA KALMA - ARTIRILDI
            'survival_bonus': 0.05  # Her adımda küçük bonus
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

        print(f"[ENV] Harita: {width}x{height}, {num_drones} drone, {num_targets} hedef")
        print(f"[ENV] Sensor: {self.sensor_range}, İletişim: {self.communication_range}, Saldırı: {self.attack_range}")

    def reset(self):
        """Ortamı sıfırla - DÜZELTİLMİŞ"""
        self.targets = []
        self.drones = []
        self.time_step = 0
        self.metrics = {k: 0 for k in self.metrics.keys()}

        # Hedefleri stratejik olarak yerleştir
        self._deploy_targets_strategically()

        # Drone'ları başlangıç pozisyonuna yerleştir
        self._deploy_drones()

        # get_observations() döndür - BU SATIR ÇOK ÖNEMLİ!
        initial_observations = self.get_observations()

        print(f"[ENV] Reset: {len(self.targets)} hedef ({self._get_target_summary()})")
        print(f"[ENV] Drone'lar: {len(self.drones)} adet")

        return initial_observations  # get_state() DEĞİL, get_observations()

    def _deploy_targets_strategically(self):
        """Hedefleri stratejik olarak yerleştir"""
        target_count = 0

        # Her hedef tipi için dağılıma göre sayı belirle
        for target_type, probability in self.target_distribution.items():
            type_count = max(1, int(self.num_targets * probability))

            for _ in range(type_count):
                if target_count >= self.num_targets:
                    break

                # Hedef tipine özel yerleşim stratejisi
                if target_type == 'tank':
                    # Tanklar savunma hattında
                    x = random.randint(self.width * 2 // 3, self.width - 100)
                    y = random.randint(100, self.height - 100)
                elif target_type == 'artillery':
                    # Topçu birimleri arkada
                    x = random.randint(self.width * 3 // 4, self.width - 50)
                    y = random.randint(150, self.height - 150)
                elif target_type == 'radar':
                    # Radar sistemleri yüksek noktalarda
                    x = random.randint(self.width // 2, self.width - 200)
                    y = random.randint(50, self.height // 3)
                elif target_type == 'aircraft':
                    # Uçaklar açık alanda
                    x = random.randint(self.width // 2, self.width - 150)
                    y = random.randint(self.height // 2, self.height - 100)
                else:  # infantry
                    # Piyadeler dağınık
                    x = random.randint(self.width // 3, self.width - 100)
                    y = random.randint(100, self.height - 100)

                self.targets.append({
                    'id': target_count,
                    'type': target_type,
                    'x': x,
                    'y': y,
                    'hp': self.target_types[target_type]['hp'],
                    'max_hp': self.target_types[target_type]['hp'],
                    'importance': self.target_types[target_type]['importance'],
                    'required_drones': self.target_types[target_type]['required_drones'],
                    'destroyed': False,
                    'detected': False,
                    'detected_by': set(),
                    'attackers': set(),  # Saldıran drone'lar
                    'damage_taken': 0
                })
                target_count += 1

        # Kalan hedefleri rastgele tip ile doldur
        while len(self.targets) < self.num_targets:
            target_type = random.choice(list(self.target_types.keys()))
            x = random.randint(self.width // 2, self.width - 100)
            y = random.randint(100, self.height - 100)

            self.targets.append({
                'id': len(self.targets),
                'type': target_type,
                'x': x,
                'y': y,
                'hp': self.target_types[target_type]['hp'],
                'max_hp': self.target_types[target_type]['hp'],
                'importance': self.target_types[target_type]['importance'],
                'required_drones': self.target_types[target_type]['required_drones'],
                'destroyed': False,
                'detected': False,
                'detected_by': set(),
                'attackers': set(),
                'damage_taken': 0
            })

    def _deploy_drones(self):
        """Drone'ları başlangıç pozisyonuna yerleştir"""
        # Drone'ları sol tarafta, stratejik olarak yerleştir
        deployment_zones = [
            (50, 100, 200, 300),  # Üst sol
            (50, 100, self.height // 2 - 100, self.height // 2 + 100),  # Orta sol
            (50, 100, self.height - 300, self.height - 100)  # Alt sol
        ]

        for i in range(self.num_drones):
            # Zone seç (round-robin)
            zone_idx = i % len(deployment_zones)
            min_x, max_x, min_y, max_y = deployment_zones[zone_idx]

            self.drones.append({
                'id': i,
                'x': random.randint(min_x, max_x),
                'y': random.randint(min_y, max_y),
                'status': 'free',  # free, engaged, attacking, destroyed
                'target_id': None,
                'team': i % 3,  # 3 takım
                'destroyed': False,
                'detected_targets': set(),
                'shared_targets': set(),
                'battery': 100.0,
                'health': 100.0,
                'total_damage': 0,
                'total_kills': 0,
                'total_movement': 0.0,
                'last_action': None
            })

    def _get_target_summary(self):
        """Hedef özeti"""
        summary = {}
        for target in self.targets:
            ttype = target['type']
            summary[ttype] = summary.get(ttype, 0) + 1
        return ', '.join([f'{k}:{v}' for k, v in summary.items()])

    def get_state(self):
        """Tüm ortam durumunu döndür"""
        # Basitleştirilmiş global state (görselleştirme için)
        return {
            'targets': [t.copy() for t in self.targets],
            'drones': [d.copy() for d in self.drones],
            'time_step': self.time_step,
            'metrics': self.metrics.copy()
        }

    def get_observations(self):
        """Her drone için lokal gözlemleri döndür"""
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
                'health': drone['health'] / 100.0,  # Drone health ekledik
                'team': drone['team'] / 3.0,

                # Görünür hedefler
                'visible_targets': [],
                'visible_target_count': 0,

                # Paylaşılan hedefler (iletişim menzilindeki drone'lardan)
                'shared_targets': [],
                'shared_target_count': 0,

                # Kendi takımındaki drone'lar
                'teammates': [],
                'teammate_count': 0,

                # Atanmış hedef (eğer varsa)
                'assigned_target': -1,
                'target_distance': 1.0,  # normalized
                'target_importance': 0.0
            }

            # Görünür hedefleri bul
            visible_targets = []
            for target in self.targets:
                if target['destroyed']:
                    continue

                dist = self._calculate_distance(drone['x'], drone['y'], target['x'], target['y'])

                if dist <= self.sensor_range:
                    visible_targets.append({
                        'id': target['id'],
                        'type': self._encode_target_type(target['type']),
                        'distance': dist / self.sensor_range,  # normalized
                        'direction_x': (target['x'] - drone['x']) / self.sensor_range,
                        'direction_y': (target['y'] - drone['y']) / self.sensor_range,
                        'importance': target['importance'] / 15.0,  # normalized
                        'hp': target['hp'] / target['max_hp'],
                        'required_drones': target['required_drones'] / 3.0,
                        'attackers': len(target['attackers']) / target['required_drones']
                    })

            obs['visible_targets'] = visible_targets
            obs['visible_target_count'] = len(visible_targets) / 10.0  # normalize

            # İletişim menzilindeki diğer drone'lardan bilgi al
            shared_targets = set()
            teammates = []

            for other_drone in self.drones:
                if other_drone['id'] == drone['id'] or other_drone['destroyed']:
                    continue

                dist = self._calculate_distance(drone['x'], drone['y'],
                                                other_drone['x'], other_drone['y'])

                if dist <= self.communication_range:
                    # Hedef bilgilerini paylaş
                    shared_targets.update(other_drone['detected_targets'])

                    # Takım arkadaşı bilgisi
                    if other_drone['team'] == drone['team']:
                        teammates.append({
                            'id': other_drone['id'],
                            'distance': dist / self.communication_range,
                            'direction_x': (other_drone['x'] - drone['x']) / self.communication_range,
                            'direction_y': (other_drone['y'] - drone['y']) / self.communication_range,
                            'status': self._encode_status(other_drone['status']),
                            'battery': other_drone['battery'] / 100.0,
                            'health': other_drone['health'] / 100.0
                        })

            obs['shared_targets'] = list(shared_targets)[:10]  # İlk 10'u
            obs['shared_target_count'] = len(shared_targets) / 10.0
            obs['teammates'] = teammates
            obs['teammate_count'] = len(teammates) / (self.num_drones // 3)

            # Atanmış hedef bilgisi
            if drone['target_id'] is not None:
                target = next((t for t in self.targets if t['id'] == drone['target_id']), None)
                if target and not target['destroyed']:
                    obs['assigned_target'] = target['id']
                    dist = self._calculate_distance(drone['x'], drone['y'], target['x'], target['y'])
                    obs['target_distance'] = dist / self.sensor_range
                    obs['target_importance'] = target['importance'] / 15.0

            observations.append(obs)

        return observations

    def _get_dead_drone_observation(self, drone_id):
        """Yok edilmiş drone için gözlem"""
        return {
            'drone_id': drone_id,
            'position': [0, 0],
            'status': 0,  # dead
            'battery': 0,
            'health': 0,
            'team': 0,
            'visible_targets': [],
            'visible_target_count': 0,
            'shared_targets': [],
            'shared_target_count': 0,
            'teammates': [],
            'teammate_count': 0,
            'assigned_target': -1,
            'target_distance': 1.0,
            'target_importance': 0.0
        }

    def _encode_status(self, status):
        """Durumu sayısal değere çevir"""
        status_map = {'free': 0.25, 'engaged': 0.5, 'attacking': 0.75, 'destroyed': 0.0}
        return status_map.get(status, 0.0)

    def _encode_target_type(self, target_type):
        """Hedef tipini sayısal değere çevir"""
        type_map = {'infantry': 0.2, 'radar': 0.4, 'artillery': 0.6, 'aircraft': 0.8, 'tank': 1.0}
        return type_map.get(target_type, 0.0)

    def _calculate_distance(self, x1, y1, x2, y2):
        """İki nokta arası mesafe"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def step(self, actions):
        """
        Ortam adımı
        actions: her drone için [move_x, move_y, attack_flag, target_id]
        move_x, move_y: -1.0 ile 1.0 arası normalized hareket
        attack_flag: 0 veya 1 (saldırı yap)
        target_id: saldırılacak hedef ID (-1 ise hedef yok)
        """
        rewards = [0.0] * len(self.drones)
        infos = [{} for _ in range(len(self.drones))]

        # 1. Drone'ları hareket ettir
        self._move_drones(actions, rewards)

        # 2. Hedef keşfi
        self._update_detection(rewards)

        # 3. Saldırı işlemleri
        self._process_attacks(actions, rewards)

        # 4. Drone durumlarını güncelle
        self._update_drone_status()

        # 5. Ödülleri hesapla (ek ödüller)
        self._calculate_additional_rewards(rewards)

        # 6. Bitti mi kontrol et
        done = self._check_done()

        # 7. Bilgi topla
        global_info = self._get_global_info()

        self.time_step += 1

        return self.get_observations(), rewards, done, global_info

    def _move_drones(self, actions, rewards):
        """Drone'ları hareket ettir"""
        for i, drone in enumerate(self.drones):
            if drone['destroyed']:
                continue

            if i < len(actions):
                action = actions[i]

                # Hareket vektörünü çıkar
                if len(action) >= 2:
                    move_x, move_y = float(action[0]), float(action[1])

                    # Clamp values
                    move_x = max(-1.0, min(1.0, move_x))
                    move_y = max(-1.0, min(1.0, move_y))

                    # Hız ile çarp
                    delta_x = move_x * self.drone_speed
                    delta_y = move_y * self.drone_speed

                    # Hareket miktarını kaydet
                    movement = math.sqrt(delta_x ** 2 + delta_y ** 2)
                    drone['total_movement'] += movement
                    self.metrics['total_movement'] += movement

                    # Konumu güncelle
                    new_x = drone['x'] + delta_x
                    new_y = drone['y'] + delta_y

                    # Sınırları kontrol et
                    drone['x'] = max(0, min(new_x, self.width))
                    drone['y'] = max(0, min(new_y, self.height))

                    # Küçük hareket cezası (enerji tüketimi)
                    rewards[i] += self.reward_params['movement_penalty'] * movement

                    # Batarya tüketimi (hareket edince)
                    drone['battery'] = max(0.0, drone['battery'] - 0.05)

            # Batarya cezası
            if drone['battery'] <= 0:
                drone['destroyed'] = True
                drone['status'] = 'destroyed'
                rewards[i] += self.reward_params['death_penalty']

    def _update_detection(self, rewards):
        """Hedef keşfini güncelle"""
        for drone in self.drones:
            if drone['destroyed']:
                continue

            prev_detected = drone['detected_targets'].copy()
            drone['detected_targets'].clear()

            for target in self.targets:
                if target['destroyed']:
                    continue

                dist = self._calculate_distance(drone['x'], drone['y'], target['x'], target['y'])

                if dist <= self.sensor_range:
                    drone['detected_targets'].add(target['id'])

                    if drone['id'] not in target['detected_by']:
                        target['detected_by'].add(drone['id'])
                        target['detected'] = True

                        # Keşif ödülü
                        rewards[drone['id']] += self.reward_params['discovery_reward']
                        self.metrics['total_discoveries'] += 1

            # Yeni keşifler için ek ödül
            new_discoveries = drone['detected_targets'] - prev_detected
            if new_discoveries:
                rewards[drone['id']] += len(new_discoveries) * self.reward_params['discovery_reward'] * 0.5

    def _process_attacks(self, actions, rewards):
        """Saldırı işlemlerini gerçekleştir"""
        for i, drone in enumerate(self.drones):
            if drone['destroyed'] or i >= len(actions):
                continue

            action = actions[i]
            if len(action) >= 4 and action[2] > 0.5:  # Saldırı flag'i
                target_id = int(action[3])

                # Hedefi bul
                target = next((t for t in self.targets if t['id'] == target_id and not t['destroyed']), None)

                if target:
                    # Mesafeyi kontrol et
                    dist = self._calculate_distance(drone['x'], drone['y'], target['x'], target['y'])

                    if dist <= self.attack_range:
                        # Hasar ver
                        damage = 1.0
                        target['hp'] -= damage
                        target['damage_taken'] += damage
                        drone['total_damage'] += damage
                        self.metrics['total_damage'] += damage

                        # Saldıran drone'lar listesine ekle
                        target['attackers'].add(drone['id'])
                        drone['status'] = 'attacking'
                        drone['target_id'] = target_id

                        # Hasar verme ödülü
                        rewards[i] += self.reward_params['damage_reward']

                        # Hedef yok edildi mi?
                        if target['hp'] <= 0:
                            target['destroyed'] = True
                            target['hp'] = 0
                            drone['total_kills'] += 1
                            self.metrics['total_destroyed'] += 1

                            # Yok etme ödülü
                            destroy_reward = target['importance'] * self.reward_params['destroy_reward_multiplier']
                            rewards[i] += destroy_reward

                            # Tüm saldıran drone'lara ödül dağıt
                            for attacker_id in target['attackers']:
                                if attacker_id < len(rewards):
                                    # Ana saldıran daha çok ödül alsın
                                    if attacker_id == drone['id']:
                                        rewards[attacker_id] += destroy_reward * 0.7
                                    else:
                                        rewards[attacker_id] += destroy_reward * 0.3

                            # Takım çalışması bonusu
                            if len(target['attackers']) >= target['required_drones']:
                                teamwork_bonus = self.reward_params['teamwork_bonus'] * len(target['attackers'])
                                for attacker_id in target['attackers']:
                                    if attacker_id < len(rewards):
                                        rewards[attacker_id] += teamwork_bonus

                            print(f"[ENV] Drone {drone['id']} hedef {target_id} ({target['type']}) yok etti!")

                    else:
                        # Hedefe yaklaşma ödülü
                        proximity_reward = max(0, 1 - dist / self.sensor_range) * self.reward_params['proximity_reward']
                        rewards[i] += proximity_reward

                        # Hedefe yönelme
                        drone['status'] = 'engaged'
                        drone['target_id'] = target_id

    def _update_drone_status(self):
        """Drone durumlarını güncelle"""
        for drone in self.drones:
            if drone['destroyed']:
                continue

            # Eğer hedef yok edildiyse veya yoksa durumu güncelle
            if drone['target_id'] is not None:
                target = next((t for t in self.targets if t['id'] == drone['target_id']), None)
                if target is None or target['destroyed']:
                    drone['status'] = 'free'
                    drone['target_id'] = None

    def _calculate_additional_rewards(self, rewards):
        """Ek ödülleri hesapla"""
        for i, drone in enumerate(self.drones):
            if drone['destroyed']:
                continue

            # Hayatta kalma bonusu
            rewards[i] += self.reward_params['survival_bonus']

            # Batarya durumuna göre küçük ceza/ödül
            if drone['battery'] < 20:
                rewards[i] += self.reward_params['battery_penalty']

            # Verimlilik bonusu (çok hasar veren drone)
            if drone['total_damage'] > 0:
                efficiency = drone['total_kills'] / drone['total_damage'] if drone['total_damage'] > 0 else 0
                rewards[i] += efficiency * self.reward_params['efficiency_bonus']

    def _check_done(self):
        """Episode bitti mi?"""
        # Tüm hedefler yok edildi
        if all(t['destroyed'] for t in self.targets):
            return True

        # Tüm drone'lar yok edildi
        if all(d['destroyed'] for d in self.drones):
            return True

        # Maksimum adım sayısına ulaşıldı
        if self.time_step >= self.max_steps:
            return True

        return False

    def _get_global_info(self):
        """Global bilgi sözlüğü"""
        destroyed_targets = sum(1 for t in self.targets if t['destroyed'])
        destroyed_drones = sum(1 for d in self.drones if d['destroyed'])

        # Hedef tipi bazlı istatistikler
        target_stats = {}
        for target_type in self.target_types.keys():
            total = sum(1 for t in self.targets if t['type'] == target_type)
            destroyed = sum(1 for t in self.targets if t['type'] == target_type and t['destroyed'])
            target_stats[f'{target_type}_total'] = total
            target_stats[f'{target_type}_destroyed'] = destroyed
            target_stats[f'{target_type}_rate'] = (destroyed / total * 100) if total > 0 else 0

        # Drone istatistikleri
        drone_stats = {
            'free_drones': sum(1 for d in self.drones if d['status'] == 'free' and not d['destroyed']),
            'engaged_drones': sum(1 for d in self.drones if d['status'] == 'engaged' and not d['destroyed']),
            'attacking_drones': sum(1 for d in self.drones if d['status'] == 'attacking' and not d['destroyed']),
            'destroyed_drones': destroyed_drones,
            'avg_battery': np.mean([d['battery'] for d in self.drones if not d['destroyed']]) if any(
                not d['destroyed'] for d in self.drones) else 0,
            'avg_damage': np.mean([d['total_damage'] for d in self.drones]) if self.drones else 0,
            'total_kills': sum(d['total_kills'] for d in self.drones)
        }

        # Başarı oranı
        success_rate = (destroyed_targets / len(self.targets)) * 100 if self.targets else 0

        # Önem ağırlıklı başarı
        weighted_success = sum(t['importance'] for t in self.targets if t['destroyed'])
        total_importance = sum(t['importance'] for t in self.targets)
        weighted_rate = (weighted_success / total_importance * 100) if total_importance > 0 else 0

        info = {
            'time_step': self.time_step,
            'episode_length': self.time_step,
            'destroyed_targets': destroyed_targets,
            'total_targets': len(self.targets),
            'success_rate': success_rate,
            'weighted_success_rate': weighted_rate,
            **self.metrics,
            **target_stats,
            **drone_stats
        }

        return info

    def render(self, mode='human'):
        """Ortamı görselleştir"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Sol grafik: Harita
        ax1 = axes[0]
        ax1.clear()
        ax1.set_xlim(0, self.width)
        ax1.set_ylim(0, self.height)
        ax1.set_title('Harp Sahası')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)

        # Hedefleri çiz
        for target in self.targets:
            color = self.target_types[target['type']]['color']
            radius = self.target_types[target['type']]['radius']

            if target['destroyed']:
                ax1.plot(target['x'], target['y'], 'x', color='red', markersize=15, markeredgewidth=2)
            else:
                circle = plt.Circle((target['x'], target['y']), radius,
                                    color=color, alpha=0.7 if target['detected'] else 0.3)
                ax1.add_patch(circle)

                # HP bar
                if target['hp'] < target['max_hp']:
                    hp_ratio = target['hp'] / target['max_hp']
                    ax1.plot([target['x'] - radius, target['x'] - radius + 2 * radius * hp_ratio],
                             [target['y'] + radius + 5, target['y'] + radius + 5],
                             'limegreen', linewidth=3)

                # Hedef tipi yazısı
                ax1.text(target['x'], target['y'] + radius + 15,
                         target['type'][:3], ha='center', va='bottom', fontsize=8)

        # Drone'ları çiz
        for drone in self.drones:
            color = ['blue', 'green', 'purple'][drone['team']]

            if drone['destroyed']:
                ax1.plot(drone['x'], drone['y'], 's', color='black', markersize=10)
            else:
                ax1.plot(drone['x'], drone['y'], 'o', color=color, markersize=8)

                # Sensor menzili
                circle = plt.Circle((drone['x'], drone['y']), self.sensor_range,
                                    color=color, alpha=0.1, linestyle='--', fill=False)
                ax1.add_patch(circle)

                # Drone ID
                ax1.text(drone['x'], drone['y'] - 15, f'D{drone["id"]}',
                         ha='center', va='top', fontsize=8)

                # Batarya
                if drone['battery'] < 30:
                    ax1.text(drone['x'], drone['y'] - 30, f'{drone["battery"]:.0f}%',
                             ha='center', va='top', fontsize=7, color='red')

        # Sağ grafik: Durum bilgisi
        ax2 = axes[1]
        ax2.clear()
        ax2.axis('off')

        info_text = f"Zaman Adımı: {self.time_step}\n"
        info_text += f"Maksimum Adım: {self.max_steps}\n\n"

        info_text += "DRONE DURUMLARI:\n"
        for drone in self.drones[:10]:  # İlk 10 drone'u göster
            status_symbol = {'free': '🟢', 'engaged': '🟡', 'attacking': '🔴', 'destroyed': '⚫'}.get(drone['status'], '⚫')
            info_text += f"D{drone['id']}: {status_symbol} Batarya: {drone['battery']:.0f}% HP: {drone['health']:.0f}%\n"

        info_text += f"\nHEDEF DURUMLARI:\n"
        destroyed = sum(1 for t in self.targets if t['destroyed'])
        info_text += f"Yok edilen: {destroyed}/{len(self.targets)}\n"

        for target_type in self.target_types.keys():
            total = sum(1 for t in self.targets if t['type'] == target_type)
            destroyed = sum(1 for t in self.targets if t['type'] == target_type and t['destroyed'])
            if total > 0:
                info_text += f"{target_type[:8]:8s}: {destroyed}/{total}\n"

        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()
        plt.pause(0.01)

        if mode == 'human':
            plt.show(block=False)
        elif mode == 'rgb_array':
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return image

        return fig
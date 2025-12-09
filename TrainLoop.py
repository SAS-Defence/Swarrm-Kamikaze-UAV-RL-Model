import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
import random
import copy
from typing import List, Dict, Tuple, Any


class ImprovedA2CAgent(nn.Module):
    """Geliştirilmiş A2C Agent - Koordinasyon desteği"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Daha derin özellik çıkarıcı
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Actor (politika)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )

        # Critic (değer)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Log standart sapması
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def forward(self, state):
        features = self.feature_extractor(state)

        # Actor output
        actor_output = self.actor(features)
        action_mean = torch.tanh(actor_output)

        # Critic output
        value = self.critic(features)

        return action_mean, value

    def get_action(self, state, epsilon=0.1):
        """Geliştirilmiş aksiyon seçimi"""
        # RANDOM AKSİYON
        if random.random() < epsilon:
            move_x = random.uniform(-1, 1)
            move_y = random.uniform(-1, 1)
            attack = random.choice([0, 1])
            target_id = -1
            return [move_x, move_y, attack, target_id], 0.0, None, None

        # POLİCY'DEN AKSİYON
        with torch.no_grad():
            action_mean, value = self.forward(state)

            # Stochastic sampling
            std = torch.exp(self.log_std).clamp(0.01, 1.0)
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()

            # Hareket
            move_x = torch.tanh(action_sample[0, 0]).item()
            move_y = torch.tanh(action_sample[0, 1]).item()

            # Saldırı
            attack_prob = torch.sigmoid(action_sample[0, 2]).item()
            attack = 1 if random.random() < attack_prob else 0

            target_id = -1

            # Log prob
            # sum(-1) sonrası [Batch] olabilir. Tekli state için [1] veya [] olabilir.
            # Bunu kesinlikle skaler yapıp liste/tensor karmaşasını önlüyoruz.
            log_prob_val = dist.log_prob(action_sample).sum(-1)
            
            # Eğer tensor ise
            if isinstance(log_prob_val, torch.Tensor):
                log_prob = log_prob_val.detach()
            else:
                log_prob = torch.tensor(log_prob_val).detach()

            return [move_x, move_y, attack, target_id], value.item(), log_prob, action_mean


class HierarchicalSwarmTrainer:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TRAINER] Device: {self.device}")

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.1)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.tau = config.get('tau', 0.001)

        # Exploration
        self.epsilon = self.epsilon_start

        # GENİŞLETİLMİŞ State boyutu (45 dim)
        self.state_dim = 45  # Koordinasyon bilgileri eklendi
        self.action_dim = 4

        # Her drone için agent
        self.agents = nn.ModuleList([
            ImprovedA2CAgent(self.state_dim, self.action_dim, hidden_dim=256).to(self.device)
            for _ in range(env.num_drones)
        ])

        # Target networks
        self.target_agents = nn.ModuleList([
            ImprovedA2CAgent(self.state_dim, self.action_dim, hidden_dim=256).to(self.device)
            for _ in range(env.num_drones)
        ])

        for target, source in zip(self.target_agents, self.agents):
            target.load_state_dict(source.state_dict())

        # Optimizers
        self.optimizers = [
            optim.AdamW(agent.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            for agent in self.agents
        ]

        # Replay buffer
        self.buffer_size = config.get('buffer_size', 10000)
        self.batch_size = config.get('batch_size', 64)
        self.replay_buffers = [deque(maxlen=self.buffer_size) for _ in range(env.num_drones)]

        # Training history
        self.history = defaultdict(list)
        self.episode = 0
        self.total_steps = 0

        # Reward tracking
        self.episode_rewards = []
        self.recent_success_rates = deque(maxlen=20)

        print(f"[TRAINER] {env.num_drones} drone için koordinasyonlu agent oluşturuldu")
        print(f"[TRAINER] State dim: {self.state_dim} (koordinasyon bilgileri dahil)")

    def _process_observation(self, observation, directive=None):
        """
        GENİŞLETİLMİŞ state representation - 45 boyut

        [0-5]   Temel: pos_x, pos_y, battery, health, status, team
        [6-8]   Sayaçlar: visible_count, shared_count, teammate_count
        [9-23]  En iyi 3 hedef (her biri 5 dim)
        [24-29] Takım bilgisi: closest teammate + avg stats
        [30-32] Saldırı koordinasyonu
        [33-34] Spatial awareness
        [35-44] KOORDİNASYON BİLGİSİ (YENİ):
                - assigned_target_id (normalized)
                - target_priority (0-1)
                - coordination_active (0/1)
                - teammate_count_on_target
                - role (scout=0, attacker=1)
                - distance_to_assigned_target
                - assigned_target_hp_ratio
                - team_progress (0-1, kaç drone gerekiyordu / kaç var)
                - should_attack_flag (0/1)
                - mission_importance (0-1)
        """
        state = torch.zeros(self.state_dim)

        if not isinstance(observation, dict):
            return state.unsqueeze(0).to(self.device)

        try:
            # Temel bilgiler [0-8]
            if 'position' in observation:
                state[0] = float(observation['position'][0])
                state[1] = float(observation['position'][1])

            state[2] = float(observation.get('battery', 0.5))
            state[3] = float(observation.get('health', 1.0))
            state[4] = float(observation.get('status', 0.0))
            state[5] = float(observation.get('team', 0.0))

            state[6] = min(float(observation.get('visible_target_count', 0.0)), 1.0)
            state[7] = min(float(observation.get('shared_target_count', 0.0)), 1.0)
            state[8] = float(observation.get('teammate_count', 0.0))

            # EN İYİ 3 HEDEFİ ÇIKAR [9-23]
            visible_targets = observation.get('visible_targets', [])
            if isinstance(visible_targets, list) and len(visible_targets) > 0:
                def target_priority(t):
                    imp = float(t.get('importance', 0.0))
                    dist = float(t.get('distance', 1.0))
                    hp = float(t.get('hp', 1.0))
                    attackers = float(t.get('attackers', 0.0))
                    required = max(float(t.get('required_drones', 1.0)), 1.0)

                    priority = imp * (1.0 - dist) * (1.0 - hp) * (1.0 - attackers / required)
                    return priority

                sorted_targets = sorted(visible_targets, key=target_priority, reverse=True)

                for idx in range(min(3, len(sorted_targets))):
                    t = sorted_targets[idx]
                    base_idx = 9 + idx * 5

                    dist_norm = float(t.get('distance', 1.0))
                    state[base_idx] = max(0.0, 1.0 - dist_norm)
                    state[base_idx + 1] = float(t.get('importance', 0.0))
                    state[base_idx + 2] = float(t.get('direction_x', 0.0))
                    state[base_idx + 3] = float(t.get('direction_y', 0.0))
                    state[base_idx + 4] = float(t.get('hp', 1.0))

            # TAKIM ARKADAŞI BİLGİSİ [24-29]
            teammates = observation.get('teammates', [])
            if isinstance(teammates, list) and len(teammates) > 0:
                closest = min(teammates, key=lambda t: t.get('distance', 1.0))
                state[24] = float(closest.get('distance', 1.0))
                state[25] = float(closest.get('direction_x', 0.0))
                state[26] = float(closest.get('direction_y', 0.0))

                avg_battery = np.mean([t.get('battery', 0.5) for t in teammates])
                avg_health = np.mean([t.get('health', 1.0) for t in teammates])
                avg_status = np.mean([t.get('status', 0.0) for t in teammates])

                state[27] = float(avg_battery)
                state[28] = float(avg_health)
                state[29] = float(avg_status)

            # SALDIRI KOORDİNASYONU [30-32]
            if len(visible_targets) > 0:
                best_target = visible_targets[0]
                attackers = float(best_target.get('attackers', 0.0))
                required = max(float(best_target.get('required_drones', 1.0)), 1.0)

                state[30] = attackers / 3.0
                state[31] = required / 3.0
                state[32] = min(attackers / required, 1.0)

            # SPATIAL AWARENESS [33-34]
            pos_x = state[0].item()
            pos_y = state[1].item()

            center_x, center_y = 0.5, 0.5
            dist_to_center = np.sqrt((pos_x - center_x) ** 2 + (pos_y - center_y) ** 2)
            state[33] = min(dist_to_center, 1.0)

            dist_to_edge = min(pos_x, pos_y, 1.0 - pos_x, 1.0 - pos_y)
            state[34] = dist_to_edge

            # 🎯 KOORDİNASYON BİLGİSİ [35-44] - YENİ!
            if directive is not None:
                # Atanmış hedef ID (normalize)
                target_id = directive.get('target_id', -1)
                state[35] = (target_id + 1) / 20.0  # 0-1 arası normalize

                # Hedef önceliği
                state[36] = directive.get('priority', 0.0) / 100.0  # 0-1 arası

                # Koordinasyon aktif mi?
                state[37] = 1.0 if directive.get('coordination', False) else 0.0

                # Aynı hedefe giden takım arkadaşı sayısı
                teammates_on_target = len(directive.get('teammates', []))
                state[38] = teammates_on_target / 3.0

                # Rol (scout=0, attacker=1)
                role = directive.get('role', 'scout')
                state[39] = 1.0 if role == 'attacker' else 0.0

                # Atanmış hedefe mesafe (eğer atanmışsa)
                # Atanmış hedefe mesafe ve YÖN (GPS verisi)
                # Artık visible listesinde aramıyoruz, direkt observation'dan alıyoruz.
                dist_gps = float(observation.get('target_distance', 1.0))
                dir_x_gps = float(observation.get('target_direction_x', 0.0))
                dir_y_gps = float(observation.get('target_direction_y', 0.0))
                
                # State'e işle [40-42]
                state[40] = dist_gps
                state[41] = dir_x_gps  # Eski HP yerine Yön X
                state[42] = dir_y_gps  # Eski Progress yerine Yön Y

                # Saldırı bayrağı
                state[43] = 1.0 if directive.get('should_attack', False) else 0.0

                # Görev önemi (priority normalizedile)
                state[44] = directive.get('priority', 0.0) / 100.0

        except Exception as e:
            print(f"[TRAINER] Observation processing error: {e}")

        return state.unsqueeze(0).to(self.device)

    def train_episode(self, coordinator=None):
        """
        KOORDİNASYONLU episode eğitimi

        coordinator: SwarmCoordinator instance (opsiyonel)
        """
        try:
            observations = self.env.reset()

            # Koordinatörü sıfırla
            if coordinator:
                coordinator.reset()

            episode_rewards = [0.0] * len(self.env.drones)

            done = False
            step_count = 0

            while not done and step_count < self.env.max_steps:
                actions = []
                states = []
                values = []
                log_probs = []

                # 🎯 KOORDİNATÖRDEN STRATEJİK DİREKTİFLER AL
                directives = None
                if coordinator:
                    directives = coordinator.get_strategic_actions(observations)

                # Her drone için aksiyon seç
                for drone_id, obs in enumerate(observations):
                    if not isinstance(obs, dict):
                        actions.append([0.0, 0.0, 0, -1])
                        states.append(None)
                        values.append(0.0)
                        log_probs.append(None)
                        continue

                    # Drone öldüyse
                    if obs.get('health', 1.0) <= 0 or obs.get('battery', 0) <= 0:
                        actions.append([0.0, 0.0, 0, -1])
                        states.append(None)
                        values.append(0.0)
                        log_probs.append(None)
                        continue

                    # Direktif al
                    directive = directives[drone_id] if directives else None

                    # State'i hazırla (direktif dahil)
                    state = self._process_observation(obs, directive)
                    states.append(state)

                    # Policy'den aksiyon al
                    action, value, log_prob, _ = self.agents[drone_id].get_action(
                        state, self.epsilon
                    )

                    values.append(value)
                    log_probs.append(log_prob)

                    move_x = float(action[0])
                    move_y = float(action[1])
                    # DÜZELTME: int(0.99) = 0 olduğu için drone asla ateş edemiyordu!
                    # Artık 0.0'dan büyükse ateş et (> 0.5 sigmoid/tanh için güvenli eşik)
                    attack = 1 if action[2] > 0.0 else 0
                    target_id = -1

                    # 🎯 DİREKTİFE GÖRE HEDEF SEÇ
                    if directive and directive.get('target_id', -1) >= 0:
                        target_id = directive['target_id']

                        # Koordinatör saldırı diyor mu?
                        if directive.get('should_attack', False):
                            attack = 1
                    else:
                        # Direktif yoksa, eski yöntemle seç
                        if attack == 1:
                            visible = obs.get('visible_targets', [])
                            if isinstance(visible, list) and len(visible) > 0:
                                def target_priority(t):
                                    imp = float(t.get('importance', 0.0))
                                    dist = float(t.get('distance', 1.0))
                                    hp = float(t.get('hp', 1.0))
                                    return imp * (1.0 - dist) * (1.0 - hp)

                                best = max(visible, key=target_priority)
                                target_id = int(best.get('id', -1))

                    actions.append([move_x, move_y, attack, target_id])

                # Ortam adımı
                next_observations, rewards, done, info = self.env.step(actions)

                # REWARD SHAPING + KOORDİNASYON ÖDÜLLERİ
                shaped_rewards = self._shape_rewards(observations, actions, rewards, next_observations, coordinator)

                # Replay buffer'a ekle
                for drone_id in range(len(self.env.drones)):
                    if drone_id < len(states) and states[drone_id] is not None:
                        # Bir sonraki state için de direktif al
                        next_directive = directives[drone_id] if directives else None
                        next_state = self._process_observation(next_observations[drone_id], next_directive)

                        self.replay_buffers[drone_id].append({
                            'state': states[drone_id],
                            'action': torch.tensor(actions[drone_id]).to(self.device),
                            'reward': torch.tensor(shaped_rewards[drone_id]).to(self.device),
                            'next_state': next_state,
                            'done': torch.tensor(done).to(self.device),
                            # Log prob: Her zaman 0-D skaler tensor olarak sakla
                            'log_prob': log_probs[drone_id].reshape(()) if log_probs[drone_id] is not None else torch.tensor(0.0).to(self.device),
                            'value': torch.tensor(values[drone_id]).to(self.device)
                        })

                        episode_rewards[drone_id] += shaped_rewards[drone_id]

                # Öğrenme adımı
                if step_count % 2 == 0 and step_count > self.batch_size:
                    self._learn()

                observations = next_observations
                step_count += 1
                self.total_steps += 1

            # Episode sonu
            total_reward = sum(episode_rewards) / max(len(episode_rewards), 1)

            # Exploration'ı azalt
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # History'ye kaydet
            self.history['episode_rewards'].append(total_reward)
            self.history['episode_length'].append(step_count)
            self.history['epsilon'].append(self.epsilon)

            # Başarı oranını kaydet
            if 'success_rate' in info:
                self.history['success_rate'].append(info['success_rate'])
                self.recent_success_rates.append(info['success_rate'])

            # 📊 DETAYLI METRİKLERİ KAYDET (YENİ)
            # Info sözlüğündeki _rate, _destroyed, avg_ gibi metrikleri history'e ekle
            for key, value in info.items():
                if any(x in key for x in ['_rate', '_destroyed', 'avg_', 'total_']):
                    # Zaten eklenmiş olanları atla (episode_rewards vs hariç)
                    if key not in ['success_rate']:
                        self.history.setdefault(key, []).append(value)

            self.episode += 1

            # İlerleme raporu
            if self.episode % 10 == 0:
                avg_success = np.mean(self.recent_success_rates) if self.recent_success_rates else 0

                # Koordinatör özeti
                coord_summary = ""
                if coordinator:
                    summary = coordinator.get_mission_summary()
                    coord_summary = f", Atamalar={summary['active_assignments']}, Serbest={summary['idle_drones']}"

                print(f"[TRAINER] Episode {self.episode}: Ödül={total_reward:.2f}, "
                      f"Adım={step_count}, ε={self.epsilon:.3f}, "
                      f"Başarı={info.get('success_rate', 0):.1f}% (Avg={avg_success:.1f}%){coord_summary}")

            return total_reward, info

        except Exception as e:
            print(f"[TRAINER] Episode eğitimi sırasında hata: {e}")
            import traceback
            traceback.print_exc()
            return 0, {'success_rate': 0}

    def _shape_rewards(self, observations, actions, base_rewards, next_observations, coordinator=None):
        """Geliştirilmiş reward shaping + koordinasyon ödülleri"""
        shaped_rewards = list(base_rewards)

        for drone_id in range(len(observations)):
            if not isinstance(observations[drone_id], dict):
                continue

            obs = observations[drone_id]
            next_obs = next_observations[drone_id]
            action = actions[drone_id]

            # 1. Hedefe yaklaşma ödülü
            visible = obs.get('visible_targets', [])
            next_visible = next_obs.get('visible_targets', [])

            if visible and next_visible:
                min_dist = min([t.get('distance', 1.0) for t in visible])
                next_min_dist = min([t.get('distance', 1.0) for t in next_visible])

                if next_min_dist < min_dist:
                    approach_reward = (min_dist - next_min_dist) * 1.0  # AZALTILDI (Farming engellemek için)
                    shaped_rewards[drone_id] += approach_reward

            # 1.5 GLOBAL NAVİGASYON (GPS) - YENİ!
            # Hedef görünmese bile, atanan hedefe yaklaşıyorsa ödül ver
            target_dist = obs.get('target_distance', 0)
            next_target_dist = next_obs.get('target_distance', 0)
            assigned_target = obs.get('assigned_target', -1)
            
            if assigned_target != -1 and target_dist > 0 and next_target_dist > 0:
                 # Yaklaşıyorsa ödül ver
                 if next_target_dist < target_dist:
                     # Fark * 2.0 (Teşvik)
                     # Not: target_distance normalize edilmiş (dist/sensor_range).
                     # Sensor range 200px. 1 birim fark = 200 piksel.
                     # Ufak hareketler bile algılanır.
                     # 2.0 -> 5.0 ARTIRILDI (Kullanıcı İsteği: Navigasyon Yardımı Artsın)
                     nav_reward = (target_dist - next_target_dist) * 5.0
                     shaped_rewards[drone_id] += nav_reward

            # 2. Keşif ödülü
            prev_visible_count = obs.get('visible_target_count', 0)
            new_visible_count = next_obs.get('visible_target_count', 0)

            if new_visible_count > prev_visible_count:
                shaped_rewards[drone_id] += 2.0  # ARTIRILDI

            # 3. WINGMAN FORMASYONU (KOL UÇUŞU) - YENİ!
            # Aynı hedefe giden arkadaşlarla yakın uçmayı ödüllendir
            # Bu, "Dağınıklığı" önler ve "Bulut" şeklinde varışı sağlar.
            teammates = obs.get('teammates', [])
            assigned_target = obs.get('assigned_target', -1)
            
            wingman_bonus = 0.0
            if assigned_target != -1:
                for tm in teammates:
                    # Sadece aynı hedefi paylaşan arkadaşa bak (Ve hayattaysa)
                    # Not: Teammate distance normalize edilmiş (dist / 200)
                    # 0.15 => 30 piksel (Oldukça yakın)
                    if tm.get('target_id') == assigned_target:
                        tm_dist = tm.get('distance', 1.0)
                        if tm_dist < 0.15: 
                            wingman_bonus += 0.05  # AZALTILDI (Farming engellemek için)
            
            shaped_rewards[drone_id] += wingman_bonus

            # 4. 🎯 KOORDİNASYON ÖDÜLLERİ (YENİ!)
            if coordinator:
                coord_reward = coordinator.get_coordination_reward(drone_id)
                shaped_rewards[drone_id] += coord_reward
            
            # 5. Batarya/health cezası yumuşatması
            battery = next_obs.get('battery', 100)
            health = next_obs.get('health', 100)

            if battery < 20:
                shaped_rewards[drone_id] -= 0.3
            if health < 30:
                shaped_rewards[drone_id] -= 0.2

            # 6. Hareketsizlik cezası azaltıldı
            if abs(action[0]) < 0.05 and abs(action[1]) < 0.05:
                shaped_rewards[drone_id] -= 0.05

            # REWARD SCALING (Hassasiyet Ayarı)
            # 1500 puanlık ödüller nöral ağı bozuyor (Exploding Gradients).
            # Tüm ödülleri 100'e bölerek normalize ediyoruz.
            # Kill: 1500 -> 15.0
            # Hit: 35 -> 0.35
            # Death: -15 -> -0.15
            shaped_rewards[drone_id] /= 100.0

        return shaped_rewards

    def _learn(self):
        """Öğrenme algoritması"""
        for drone_id in range(len(self.agents)):
            if len(self.replay_buffers[drone_id]) < self.batch_size:
                continue

            batch = random.sample(self.replay_buffers[drone_id], self.batch_size)

            states = torch.cat([b['state'] for b in batch])
            actions = torch.stack([b['action'] for b in batch])
            rewards = torch.stack([b['reward'] for b in batch])
            next_states = torch.cat([b['next_state'] for b in batch])
            dones = torch.stack([b['done'] for b in batch])

            try:
                # GAE
                with torch.no_grad():
                    _, next_values = self.target_agents[drone_id](next_states)
                    next_values = next_values.squeeze()
                    td_targets = rewards + self.gamma * next_values * (~dones)

                # Current values
                action_means, values = self.agents[drone_id](states)
                values = values.squeeze()

                # Advantages
                advantages = td_targets - values

                if advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 0. Eksik tanımları tamamla
                critic_loss = F.smooth_l1_loss(values, td_targets)
                std = torch.exp(self.agents[drone_id].log_std).clamp(0.01, 1.0) # std tanımla!

                # PPO LOSS CALCULATION
                # PPO LOSS CALCULATION
                # 1. Eski log_prob'ları al
                # Squeeze ile kesinlikle (Batch,) boyutunda olduğundan emin oluyoruz.
                old_log_probs = torch.stack([b['log_prob'] for b in batch]).squeeze().detach()

                # 2. Sadece MoveX, MoveY, Attack (İlk 3 boyut) üzerinden loss hesapla!
                # 4. boyut (Target ID) heuristics ile belirleniyor, network bunu öğrenmeye çalışmamalı.
                active_action_means = action_means[:, :3] 
                active_actions = actions[:, :3]
                
                # std de 4 boyutlu, onu da kes
                active_std = std[:, :3]

                dist = torch.distributions.Normal(active_action_means, active_std)
                new_log_probs = dist.log_prob(active_actions).sum(-1)

                # 3. Ratio ve Clipping
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages.detach()
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages.detach()

                actor_loss = -torch.min(surr1, surr2).mean()

                # 4. Entropy (Exploration)
                entropy = dist.entropy().mean()
                entropy_bonus = 0.03 * entropy # Exploration artırıldı (Hard Mode için)

                total_loss = critic_loss + actor_loss - entropy_bonus

                # Optimize
                self.optimizers[drone_id].zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[drone_id].parameters(), 0.5)
                self.optimizers[drone_id].step()

                # Soft update
                self._soft_update(self.agents[drone_id], self.target_agents[drone_id])

                # Logging
                if self.episode % 10 == 0:
                    self.history.setdefault(f'loss_drone_{drone_id}', []).append(total_loss.item())
                    self.history.setdefault(f'critic_loss_drone_{drone_id}', []).append(critic_loss.item())
                    self.history.setdefault(f'actor_loss_drone_{drone_id}', []).append(actor_loss.item())

            except Exception as e:
                print(f"[TRAINER] Drone {drone_id} öğrenme hatası: {e}")
                continue

    def _soft_update(self, source, target):
        """Soft update"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_model(self, path="models/"):
        """Model kaydet"""
        import os
        os.makedirs(path, exist_ok=True)

        checkpoint = {
            'agents_state_dict': [agent.state_dict() for agent in self.agents],
            'target_agents_state_dict': [agent.state_dict() for agent in self.target_agents],
            'optimizers_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
            'history': dict(self.history),
            'episode': self.episode,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'config': self.config
        }

        torch.save(checkpoint, f"{path}/model_episode_{self.episode}.pth")
        print(f"[TRAINER] Model kaydedildi: {path}/model_episode_{self.episode}.pth")

    def load_model(self, path):
        """Model yükle"""
        checkpoint = torch.load(path, map_location=self.device)

        for i, state_dict in enumerate(checkpoint['agents_state_dict']):
            self.agents[i].load_state_dict(state_dict)

        for i, state_dict in enumerate(checkpoint['target_agents_state_dict']):
            self.target_agents[i].load_state_dict(state_dict)

        for i, state_dict in enumerate(checkpoint['optimizers_state_dict']):
            self.optimizers[i].load_state_dict(state_dict)

        self.history = defaultdict(list, checkpoint['history'])
        self.episode = checkpoint['episode']
        self.total_steps = checkpoint['total_steps']
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)

        print(f"[TRAINER] Model yüklendi: {path}")
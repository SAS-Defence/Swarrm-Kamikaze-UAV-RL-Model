import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
import random
import copy
from typing import List, Dict, Tuple, Any


class A2CAgent(nn.Module):
    """Advantage Actor-Critic Agent"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Ortak özellik çıkarıcı
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Actor (politika)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )

        # Critic (değer)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Log standart sapması (stochastic politika için)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        features = self.feature_extractor(state)

        # Actor output
        actor_output = self.actor(features)
        action_mean = torch.tanh(actor_output)  # -1 ile 1 arası

        # Critic output
        value = self.critic(features)

        return action_mean, value

    def get_action(self, state, epsilon=0.1):
        """Epsilon-greedy ile aksiyon seç
        ÇIKTI: [move_x, move_y, attack, target_id]
        target_id burada -1 placeholder, gerçek hedefi trainer seçecek.
        """
        # RANDOM AKSİYON (EXPLORATION)
        if random.random() < epsilon:
            move_x = random.uniform(-1, 1)
            move_y = random.uniform(-1, 1)
            attack = random.choice([0, 1])
            target_id = -1  # gerçek hedefi train_episode içinde gözleme göre seçeceğiz
            return [move_x, move_y, attack, target_id], 0.0, None, None

        # POLİCY'DEN AKSİYON (EXPLOITATION)
        with torch.no_grad():
            action_mean, value = self.forward(state)

            # Hareket -1..1 arası
            move_x = action_mean[0, 0].item()
            move_y = action_mean[0, 1].item()

            # Saldırı olasılığı
            attack_prob = torch.sigmoid(action_mean[0, 2]).item()
            attack = 1 if random.random() < attack_prob else 0

            # target_id burada üretilmiyor, trainer gözleme göre seçiyor
            target_id = -1

            # Log prob (sadece hareket + attack için)
            action_tensor = torch.tensor([[move_x, move_y, float(attack), float(target_id)]])
            log_prob = self._get_log_prob(action_mean, action_tensor)

            return [move_x, move_y, attack, target_id], value.item(), log_prob, action_mean

    def _get_log_prob(self, mean, action):
        """Aksiyonun log olasılığını hesapla"""
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        return log_prob


class HierarchicalSwarmTrainer:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TRAINER] Device: {self.device}")

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.998)
        self.tau = config.get('tau', 0.005)  # Soft update için

        # Exploration
        self.epsilon = self.epsilon_start

        # State ve action boyutları
        self.state_dim = 10  # [x, y, battery, health, status, visible_targets]
        self.action_dim = 4  # [move_x, move_y, attack, target_id]

        # Her drone için bir agent
        self.agents = nn.ModuleList([
            A2CAgent(self.state_dim, self.action_dim).to(self.device)
            for _ in range(env.num_drones)
        ])

        # Target network'lar (stabilite için)
        self.target_agents = nn.ModuleList([
            A2CAgent(self.state_dim, self.action_dim).to(self.device)
            for _ in range(env.num_drones)
        ])

        # Target network'ları güncelle
        for target, source in zip(self.target_agents, self.agents):
            target.load_state_dict(source.state_dict())

        # Optimizer'lar
        self.optimizers = [
            optim.Adam(agent.parameters(), lr=self.learning_rate)
            for agent in self.agents
        ]

        # Replay buffer
        self.buffer_size = config.get('buffer_size', 5000)
        self.batch_size = config.get('batch_size', 32)
        self.replay_buffers = [deque(maxlen=self.buffer_size) for _ in range(env.num_drones)]

        # Training history
        self.history = defaultdict(list)
        self.episode = 0
        self.total_steps = 0

        print(f"[TRAINER] {env.num_drones} drone için {len(self.agents)} agent oluşturuldu")
        print(f"[TRAINER] State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"[TRAINER] Learning rate: {self.learning_rate}, Gamma: {self.gamma}")
        print(f"[TRAINER] Epsilon: {self.epsilon_start} → {self.epsilon_end}")

    def _process_observation(self, observation):
        """Gözlemi state vektörüne çevir

        State yapısı (10 boyut):
        0: pos_x (0-1)
        1: pos_y (0-1)
        2: battery (0-1)
        3: health (0-1)
        4: status (0-1, encode_status)
        5: visible_target_count (0-1)
        6: best_target_proximity (0-1, 1 = çok yakın)
        7: best_target_importance (0-1)
        8: best_target_dir_x (-1..1)
        9: best_target_dir_y (-1..1)
        """
        state = torch.zeros(self.state_dim)

        if not isinstance(observation, dict):
            return state.unsqueeze(0).to(self.device)

        try:
            # Pozisyon
            if 'position' in observation:
                state[0] = float(observation['position'][0])  # x
                state[1] = float(observation['position'][1])  # y

            # Batarya ve health
            state[2] = float(observation.get('battery', 0.5))
            state[3] = float(observation.get('health', 1.0))

            # Durum kodu
            state[4] = float(observation.get('status', 0.0))

            # Görülen hedef sayısı (normalize)
            visible_count = observation.get('visible_target_count', 0.0)
            state[5] = min(float(visible_count), 1.0)

            # EN ÖNEMLİ / UYGUN HEDEFİ ÇIKAR
            best_proximity = 0.0
            best_importance = 0.0
            best_dir_x = 0.0
            best_dir_y = 0.0

            visible_targets = observation.get('visible_targets', [])
            if isinstance(visible_targets, list) and len(visible_targets) > 0:
                # importance yüksek, distance küçük olanı seç
                def target_key(t):
                    imp = float(t.get('importance', 0.0))
                    dist = float(t.get('distance', 1.0))  # 0=yakın, 1=uzak
                    return (imp, - (1.0 - dist))  # önem ↑, yakınlık ↑

                best = max(visible_targets, key=target_key)

                dist_norm = float(best.get('distance', 1.0))
                best_proximity = max(0.0, 1.0 - dist_norm)  # 0-1 arası, 1 = çok yakın
                best_importance = float(best.get('importance', 0.0))
                best_dir_x = float(best.get('direction_x', 0.0))
                best_dir_y = float(best.get('direction_y', 0.0))

            state[6] = best_proximity
            state[7] = best_importance
            state[8] = best_dir_x
            state[9] = best_dir_y

        except Exception as e:
            print(f"[TRAINER] Observation processing error: {e}")

        return state.unsqueeze(0).to(self.device)

    def train_episode(self):
        """Bir episode eğit"""
        try:
            observations = self.env.reset()
            episode_rewards = []

            done = False
            step_count = 0

            while not done and step_count < min(500, self.env.max_steps):
                actions = []

                # Her drone için aksiyon seç
                for drone_id, obs in enumerate(observations):
                    if not isinstance(obs, dict):
                        actions.append([0.0, 0.0, 0, -1])
                        continue

                    # Drone öldüyse veya bataryası yoksa
                    if obs.get('health', 1.0) <= 0 or obs.get('battery', 0) <= 0:
                        actions.append([0.0, 0.0, 0, -1])
                        continue

                    # State'i hazırla
                    state = self._process_observation(obs)

                    # Policy'den aksiyon al (move_x, move_y, attack, dummy_target)
                    action, value, log_prob, action_mean = self.agents[drone_id].get_action(
                        state, self.epsilon
                    )

                    move_x = float(action[0])
                    move_y = float(action[1])
                    attack = int(action[2])  # 0 veya 1 olmalı
                    target_id = -1

                    # Eğer saldırı yapılacaksa, hedefi GÖZLEM ÜZERİNDEN seç
                    if attack == 1:
                        visible = obs.get('visible_targets', [])
                        if isinstance(visible, list) and len(visible) > 0:
                            # Önem + yakınlık kriterine göre en iyi hedef
                            def target_key(t):
                                imp = float(t.get('importance', 0.0))
                                dist = float(t.get('distance', 1.0))
                                return (imp, - (1.0 - dist))

                            best = max(visible, key=target_key)
                            target_id = int(best.get('id', -1))

                    actions.append([move_x, move_y, attack, target_id])



                    move_x = float(action[0])
                    move_y = float(action[1])
                    attack = int(action[2] > 0.5)  # 0/1'e çevir

                    # 🔥 HEDEF SEÇİMİNİ POLICY'DEN DEĞİL, GÖZLEM ÜZERİNDEN YAP
                    target_id = -1
                    if attack == 1:
                        visible = obs.get('visible_targets', [])
                        if visible:
                            # Önce önem (importance), eşitlikte en yakın (distance en küçük)
                            best = max(
                                visible,
                                key=lambda t: (t.get('importance', 0.0), -t.get('distance', 1.0))
                            )
                            target_id = int(best['id'])

                    actions.append([move_x, move_y, attack, target_id])

                # Ortam adımı
                next_observations, rewards, done, info = self.env.step(actions)

                # Replay buffer'a ekle (sadece gerekli drone'lar için)
                for drone_id in range(min(len(observations), len(actions), len(rewards))):
                    if drone_id < len(self.replay_buffers):
                        state = self._process_observation(observations[drone_id])
                        next_state = self._process_observation(next_observations[drone_id])

                        # Sadece geçerli state varsa ekle
                        if state.sum().item() != 0 or next_state.sum().item() != 0:
                            self.replay_buffers[drone_id].append({
                                'state': state,
                                'action': torch.tensor(actions[drone_id]).to(self.device),
                                'reward': torch.tensor(rewards[drone_id]).to(self.device),
                                'next_state': next_state,
                                'done': torch.tensor(done).to(self.device)
                            })

                # Öğrenme adımı
                if step_count % 4 == 0 and step_count > 0:  # Her 4 adımda bir öğren
                    self._learn()

                episode_rewards.append(sum(rewards))
                observations = next_observations
                step_count += 1
                self.total_steps += 1

            # Episode sonu işlemleri
            total_reward = sum(episode_rewards)

            # Exploration'ı azalt
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # History'ye kaydet
            self.history['episode_rewards'].append(total_reward)
            self.history['episode_length'].append(step_count)
            self.history['epsilon'].append(self.epsilon)

            # Başarı oranını kaydet
            if 'success_rate' in info:
                self.history['success_rate'].append(info['success_rate'])

            self.episode += 1

            print(f"[TRAINER] Episode {self.episode}: Ödül={total_reward:.2f}, "
                  f"Adım={step_count}, ε={self.epsilon:.3f}, "
                  f"Başarı={info.get('success_rate', 0):.1f}%")

            return total_reward, info

        except Exception as e:
            print(f"[TRAINER] Episode eğitimi sırasında hata: {e}")
            import traceback
            traceback.print_exc()
            return 0, {'success_rate': 0, 'efficiency': 0, 'coordination': 0}

    def _learn(self):
        """Replay buffer'dan öğren"""
        for drone_id in range(len(self.agents)):
            if len(self.replay_buffers[drone_id]) < self.batch_size:
                continue

            # Batch oluştur
            batch = random.sample(self.replay_buffers[drone_id], self.batch_size)

            # Batch verilerini topla
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            for b in batch:
                states.append(b['state'])
                actions.append(b['action'])
                rewards.append(b['reward'])
                next_states.append(b['next_state'])
                dones.append(b['done'])

            # Tensor'ları birleştir
            try:
                states = torch.cat(states)
                actions = torch.stack(actions)
                rewards = torch.stack(rewards)
                next_states = torch.cat(next_states)
                dones = torch.stack(dones)

                # Target Q değerlerini hesapla
                with torch.no_grad():
                    _, next_values = self.target_agents[drone_id](next_states)
                    target_values = rewards + self.gamma * next_values.squeeze() * (~dones)

                # Mevcut Q değerleri
                action_means, values = self.agents[drone_id](states)
                values = values.squeeze()

                # Advantage
                advantages = target_values - values

                # Critic loss (MSE)
                critic_loss = F.mse_loss(values, target_values)

                # Actor loss (policy gradient)
                std = torch.exp(self.agents[drone_id].log_std)
                dist = torch.distributions.Normal(action_means, std)
                log_probs = dist.log_prob(actions).sum(-1)

                # Normalize advantages
                if advantages.std() > 0:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                actor_loss = -(log_probs * advantages).mean()

                # Entropy bonus (exploration)
                entropy_bonus = 0.01 * dist.entropy().mean()

                # Toplam loss
                loss = critic_loss + actor_loss - entropy_bonus

                # Backpropagation
                self.optimizers[drone_id].zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.agents[drone_id].parameters(), 1.0)

                self.optimizers[drone_id].step()

                # Target network'ları soft update et
                self._soft_update(self.agents[drone_id], self.target_agents[drone_id])

                # Kayıt
                self.history.setdefault(f'loss_drone_{drone_id}', []).append(loss.item())
                self.history.setdefault(f'critic_loss_drone_{drone_id}', []).append(critic_loss.item())
                self.history.setdefault(f'actor_loss_drone_{drone_id}', []).append(actor_loss.item())

            except Exception as e:
                print(f"[TRAINER] Drone {drone_id} öğrenme hatası: {e}")
                continue

    def _soft_update(self, source, target):
        """Soft update: target = tau * source + (1 - tau) * target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_model(self, path="models/"):
        """Modeli kaydet"""
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
        """Modeli yükle"""
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
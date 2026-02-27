"""
Anti-Detection System for Territorial.io Bot
=============================================
Makes the bot behave more like a human to avoid detection.
"""

import logging
import random
import time
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class HumanBehaviorProfile:
    """Profile defining human-like behavior parameters."""
    # Mouse movement timing
    min_move_duration: float = 0.15  # seconds
    max_move_duration: float = 0.8
    
    # Click timing  
    min_click_interval: float = 0.3
    max_click_interval: float = 2.0
    
    # Movement patterns
    movement_smoothing: float = 0.7  # 0-1, higher = more curved paths
    
    # Reaction delays
    min_reaction_time: float = 0.2
    max_reaction_time: float = 1.5
    
    # Variation in behavior
    randomness_factor: float = 0.3  # How much to vary between sessions


class AntiDetection:
    """
    Anti-detection system to make bot behavior more human-like.
    
    Features:
    - Human-like mouse movement with curves and variations
    - Variable delays between actions
    - Random behavioral patterns to avoid predictability
    - Reaction time simulation
    """
    
    def __init__(self, profile: Optional[HumanBehaviorProfile] = None):
        self.profile = profile or HumanBehaviorProfile()
        
        # State tracking
        self.last_action_time = 0.0
        self.last_action_type = None
        self.action_sequence = []  # Track recent actions for pattern variation
        self.session_id = random.randint(1000, 9999)
        
        # Per-session variation (different "personality" each session)
        self._generate_session_personality()
        
        # Performance tracking
        self.total_actions = 0
        self.detection_risk_score = 0.0
        
        logger.info(f"AntiDetection initialized with session {self.session_id}")
    
    def _generate_session_personality(self):
        """Generate a unique behavioral personality for this session."""
        # Randomize base parameters within reasonable ranges
        self.personality = {
            "patience": random.uniform(0.4, 0.9),
            "aggression_speed": random.uniform(0.5, 1.0),
            "movement_style": random.choice(["smooth", "jerky", "variable"]),
            "click_timing": random.uniform(0.6, 1.0),
            "reaction_speed": random.uniform(0.5, 1.0),
        }
        
        logger.debug(f"Session personality: {self.personality}")
    
    # ── Human-like Delays ────────────────────────────────────────────────────
    
    def get_action_delay(self, action_type: str = "default") -> float:
        """
        Get a human-like delay before taking an action.
        Varies based on action type and recent history.
        """
        base_delay = random.uniform(
            self.profile.min_reaction_time,
            self.profile.max_reaction_time
        )
        
        # Adjust based on personality
        base_delay *= self.personality["reaction_speed"]
        
        # Add variation based on recent actions (avoid patterns)
        if len(self.action_sequence) > 3:
            recent_actions = self.action_sequence[-3:]
            # If same action repeated, add more delay
            if len(set(recent_actions)) == 1:
                base_delay *= random.uniform(1.2, 1.8)
        
        # Reduce detection risk score
        self.detection_risk_score *= 0.95
        
        return base_delay
    
    def get_inter_action_delay(self) -> float:
        """Get delay between consecutive actions."""
        delay = random.uniform(
            self.profile.min_click_interval,
            self.profile.max_click_interval
        )
        
        # Add randomness based on personality
        delay *= self.personality["patience"]
        
        return delay
    
    # ── Human-like Mouse Movements ──────────────────────────────────────────
    
    def generate_human_path(self, start: Tuple[int, int], 
                          end: Tuple[int, int]) -> list:
        """
        Generate a human-like mouse movement path with curves and variations.
        Returns list of (x, y) intermediate points.
        """
        start_x, start_y = start
        end_x, end_y = end
        
        # Calculate direct distance
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Number of intermediate points based on distance
        num_points = max(3, int(distance / 30))
        
        # Generate base linear path
        t = np.linspace(0, 1, num_points)
        base_x = start_x + (end_x - start_x) * t
        base_y = start_y + (end_y - start_y) * t
        
        # Add curvature based on movement style
        if self.personality["movement_style"] == "smooth":
            # Bezier-like curve with slight offset
            control_offset = distance * random.uniform(-0.15, 0.15)
            curve_x = base_x + np.sin(t * np.pi) * control_offset * random.choice([-1, 1])
            curve_y = base_y + np.cos(t * np.pi) * control_offset * random.choice([-1, 1])
            path = list(zip(curve_x.astype(int), curve_y.astype(int)))
            
        elif self.personality["movement_style"] == "jerky":
            # Occasional direction changes
            path = []
            for i in range(num_points):
                if i % 3 == 0 and i > 0:
                    # Add slight random offset
                    offset_x = random.randint(-15, 15)
                    offset_y = random.randint(-15, 15)
                    px = int(base_x[i] + offset_x)
                    py = int(base_y[i] + offset_y)
                else:
                    px, py = int(base_x[i]), int(base_y[i])
                path.append((px, py))
                
        else:  # variable
            # Mix of smooth and jerky
            if random.random() > 0.5:
                # Add micro-movements (human tremor)
                noise_x = np.random.normal(0, 2, num_points)
                noise_y = np.random.normal(0, 2, num_points)
                path = list(zip(
                    (base_x + noise_x).astype(int), 
                    (base_y + noise_y).astype(int)
                ))
            else:
                path = list(zip(base_x.astype(int), base_y.astype(int)))
        
        # Ensure start and end points are exact
        if path:
            path[0] = start
            path[-1] = end
        
        return path
    
    def get_movement_duration(self, distance: float) -> float:
        """Get realistic movement duration based on distance."""
        # Human mouse movement is typically 100-500 pixels/second
        speed = random.uniform(150, 450) * self.personality["aggression_speed"]
        duration = distance / speed
        
        # Clamp to reasonable range
        return max(self.profile.min_move_duration, 
                  min(self.profile.max_move_duration, duration))
    
    # ── Behavioral Variation ──────────────────────────────────────────────────
    
    def should_vary_pattern(self, base_action: str) -> bool:
        """
        Determine if we should vary from the expected pattern.
        Adds unpredictability to avoid detection.
        """
        # Check recent action history
        if len(self.action_sequence) < 3:
            return random.random() < 0.3
        
        # If same action repeated recently, vary it
        if len(set(self.action_sequence[-3:])) == 1:
            return True
        
        # Random variation based on randomness factor
        return random.random() < self.profile.randomness_factor
    
    def get_alternative_action(self, intended_action: str) -> str:
        """Get an alternative action to add variation."""
        alternatives = {
            "ATTACK_N": ["ATTACK_NE", "ATTACK_NW", "STAY"],
            "ATTACK_S": ["ATTACK_SE", "ATTACK_SW", "STAY"], 
            "ATTACK_E": ["ATTACK_NE", "ATTACK_SE", "STAY"],
            "ATTACK_W": ["ATTACK_NW", "ATTACK_SW", "STAY"],
            "ATTACK_NE": ["ATTACK_N", "ATTACK_E", "STAY"],
            "ATTACK_NW": ["ATTACK_N", "ATTACK_W", "STAY"],
            "ATTACK_SE": ["ATTACK_S", "ATTACK_E", "STAY"],
            "ATTACK_SW": ["ATTACK_S", "ATTACK_W", "STAY"],
            "STAY": ["ATTACK_N", "ATTACK_S", "ATTACK_E", "ATTACK_W"],
        }
        
        options = alternatives.get(intended_action, [intended_action])
        return random.choice(options)
    
    def record_action(self, action_type: str):
        """Record action for pattern analysis."""
        self.action_sequence.append(action_type)
        self.total_actions += 1
        
        # Keep sequence limited
        if len(self.action_sequence) > 20:
            self.action_sequence = self.action_sequence[-20:]
        
        self.last_action_type = action_type
        self.last_action_time = time.time()
    
    # ── Detection Risk Management ─────────────────────────────────────────────
    
    def calculate_detection_risk(self) -> float:
        """Calculate current detection risk based on behavior patterns."""
        risk = 0.0
        
        # Check for suspicious patterns
        if len(self.action_sequence) >= 10:
            # Too consistent = suspicious
            unique_actions = len(set(self.action_sequence[-10:]))
            if unique_actions <= 2:
                risk += 0.3
            
            # Too fast = suspicious
            if self.total_actions > 50:
                recent_rate = len(self.action_sequence[-10:]) / 10  # actions per second
                if recent_rate > 1.5:  # More than 1.5 actions per second average
                    risk += 0.2
        
        # Update risk score with decay
        self.detection_risk_score = max(risk, self.detection_risk_score * 0.9)
        
        return self.detection_risk_score
    
    def get_safe_action_parameters(self) -> Dict:
        """
        Get parameters adjusted to reduce detection risk.
        Returns dict of safe timing and movement parameters.
        """
        risk = self.calculate_detection_risk()
        
        params = {
            "movement_duration_multiplier": 1.0 + risk * 0.5,
            "delay_multiplier": 1.0 + risk * 0.3,
            "path_complexity": 1.0 + risk,
        }
        
        # If risk is high, add extra safety measures
        if risk > 0.5:
            params["add_micro_pauses"] = True
            params["add_random_deviations"] = True
        
        return params
    
    # ── Async Support ─────────────────────────────────────────────────────────
    
    async def human_delay(self, base_duration: float = None):
        """Async-compatible delay with human-like variation."""
        if base_duration is None:
            duration = self.get_action_delay()
        else:
            duration = base_duration * random.uniform(0.8, 1.2)
        
        await asyncio.sleep(duration)
    
    # ── Strategy Variation ────────────────────────────────────────────────────
    
    def get_behavioral_modifiers(self) -> Dict:
        """
        Get behavioral modifiers based on current "personality".
        These can adjust the bot's strategic behavior.
        """
        modifiers = {
            "patience_factor": self.personality["patience"],
            "speed_factor": self.personality["aggression_speed"],
            "risk_tolerance": random.uniform(0.3, 0.7),
            "defensive_tendency": random.uniform(0.2, 0.6),
        }
        
        return modifiers

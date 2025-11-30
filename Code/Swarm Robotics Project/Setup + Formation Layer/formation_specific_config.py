"""
Formation-Specific Contingency Parameters
==========================================

Optimal contingency configurations for each formation based on dashboard analysis
"""

from dataclasses import dataclass
from contingency_reserve import ContingencyReserves
from enum import Enum


class FormationRobustness(Enum):
    """Robustness classification based on performance metrics"""
    EXCELLENT = 1
    GOOD = 2
    FAIR = 3
    POOR = 4


@dataclass
class FormationProfile:
    """Complete contingency and performance profile for each formation"""
    name: str
    converged: bool
    convergence_rate: float  # 0-1
    convergence_time_seconds: float
    mean_position_error_m: float
    energy_efficiency: float  # J/m
    min_safe_distance: float
    robustness: FormationRobustness
    contingency_reserves: ContingencyReserves
    notes: str


# ============================================================================
# FORMATION-SPECIFIC PROFILES
# ============================================================================

FORMATION_PROFILES = {
    
    'cupid': FormationProfile(
        name='CUPID',
        converged=False,  # Stalled at 100s
        convergence_rate=0.85,
        convergence_time_seconds=100.0,
        mean_position_error_m=0.039,
        energy_efficiency=0.563,  # J/m
        min_safe_distance=0.294,
        robustness=FormationRobustness.FAIR,
        contingency_reserves=ContingencyReserves(
            energy_reserve_percent=0.15,
            collision_buffer_margin=0.04,  # Slightly tighter - more stable
            max_position_error_threshold=0.15,
            convergence_timeout=12000,  # ~100s
            min_safe_distance=0.30
        ),
        notes="""
        - Moderate convergence, but gets stuck near target
        - Position errors concentrate near nominal, some outliers
        - Energy consistent around 3.1J average
        - Good for general missions, acceptable risks
        - RECOMMENDATION: Monitor convergence timeout carefully
        """
    ),
    
    'dragon': FormationProfile(
        name='DRAGON',
        converged=False,  # Does not converge reliably
        convergence_rate=0.80,
        convergence_time_seconds=100.0,
        mean_position_error_m=0.060,  # Highest error!
        energy_efficiency=0.573,  # J/m
        min_safe_distance=0.285,
        robustness=FormationRobustness.POOR,
        contingency_reserves=ContingencyReserves(
            energy_reserve_percent=0.20,  # Stricter energy buffer
            collision_buffer_margin=0.06,  # Larger collision margin
            max_position_error_threshold=0.10,  # Stricter error tolerance
            convergence_timeout=8000,  # Lower timeout - reset earlier
            min_safe_distance=0.32  # Higher safety threshold
        ),
        notes="""
        - POOREST performance metrics
        - Highest position errors (6cm mean) - unreliable
        - High inter-agent distance variability
        - Inconsistent energy consumption pattern
        - RECOMMENDATION: Use only for low-criticality missions
        - Consider parameter retuning or different formation
        - Monitor agent failures carefully
        """
    ),
    
    'flag': FormationProfile(
        name='FLAG',
        converged=False,  # Stalled but very close
        convergence_rate=0.95,  # BEST convergence rate
        convergence_time_seconds=100.0,
        mean_position_error_m=0.015,  # Good precision
        energy_efficiency=0.553,  # J/m - most efficient
        min_safe_distance=0.267,
        robustness=FormationRobustness.GOOD,
        contingency_reserves=ContingencyReserves(
            energy_reserve_percent=0.12,  # Can run tighter - very efficient
            collision_buffer_margin=0.04,
            max_position_error_threshold=0.15,
            convergence_timeout=12000,
            min_safe_distance=0.30
        ),
        notes="""
        - HIGHEST convergence rate (95%) - closest to converged
        - Best energy efficiency (0.553 J/m)
        - Very consistent inter-agent spacing
        - Excellent for precision-required tasks
        - RECOMMENDATION: Preferred for critical precision operations
        - Safe to run tighter energy reserves
        - Can push convergence longer before timeout
        """
    ),
    
    'helix': FormationProfile(
        name='HELIX',
        converged=True,  # YES - converges reliably ✓
        convergence_rate=1.0,  # 100% successful
        convergence_time_seconds=0.74,  # FASTEST convergence!
        mean_position_error_m=0.008,  # Excellent accuracy
        energy_efficiency=0.573,  # J/m
        min_safe_distance=0.295,
        robustness=FormationRobustness.EXCELLENT,
        contingency_reserves=ContingencyReserves(
            energy_reserve_percent=0.15,
            collision_buffer_margin=0.03,  # Can use minimal buffer - very robust
            max_position_error_threshold=0.12,  # Tight tolerance acceptable
            convergence_timeout=15000,  # Can give more time - rarely needed
            min_safe_distance=0.30
        ),
        notes="""
        - FASTEST CONVERGENCE (0.74s) - highly reactive
        - Perfect convergence rate (100%)
        - Excellent position accuracy (8mm mean error)
        - Most robust formation for dynamic tasks
        - RECOMMENDATION: PREFERRED for time-critical applications
        - Minimal collision margins needed - room for optimization
        - Excellent for formations requiring quick response
        """
    ),
    
    'mandala': FormationProfile(
        name='MANDALA',
        converged=True,  # YES - converges reliably ✓
        convergence_rate=1.0,  # 100% successful
        convergence_time_seconds=0.69,  # FASTEST overall!
        mean_position_error_m=0.003,  # BEST ACCURACY!
        energy_efficiency=0.563,  # J/m - efficient
        min_safe_distance=0.285,
        robustness=FormationRobustness.EXCELLENT,
        contingency_reserves=ContingencyReserves(
            energy_reserve_percent=0.15,
            collision_buffer_margin=0.03,  # Minimal buffer sufficient
            max_position_error_threshold=0.12,
            convergence_timeout=15000,
            min_safe_distance=0.30
        ),
        notes="""
        - ABSOLUTE FASTEST CONVERGENCE (0.69s)
        - PERFECT convergence rate (100%) 
        - BEST POSITION ACCURACY (3mm error) - exceptional!
        - Most reliable formation for all applications
        - RECOMMENDATION: DEFAULT CHOICE for critical missions
        - Works well with conservative contingency settings
        - Excellent stability - can push to aggressive parameters if needed
        """
    )
}


# ============================================================================
# USAGE: Contingency Configuration by Formation
# ============================================================================

def get_formation_reserves(formation_name: str) -> ContingencyReserves:
    """
    Get optimized contingency reserves for specified formation.
    
    Args:
        formation_name: 'cupid', 'dragon', 'flag', 'helix', or 'mandala'
    
    Returns:
        ContingencyReserves configured for that formation
    """
    formation_name = formation_name.lower()
    
    if formation_name not in FORMATION_PROFILES:
        raise ValueError(f"Unknown formation: {formation_name}")
    
    profile = FORMATION_PROFILES[formation_name]
    return profile.contingency_reserves


def print_formation_analysis():
    """Print comprehensive analysis of all formations with contingency impact"""
    
    print("\n" + "=" * 80)
    print("CONTINGENCY ANALYSIS: SWARM ROBOTICS FORMATIONS")
    print("=" * 80)
    
    # Rank formations
    formations_ranked = sorted(
        FORMATION_PROFILES.values(),
        key=lambda p: (p.robustness.value, -p.convergence_rate, p.mean_position_error_m)
    )
    
    print("\n📊 FORMATION RANKING (by robustness & performance):\n")
    
    for rank, profile in enumerate(formations_ranked, 1):
        status_icon = "✓" if profile.converged else "⚠"
        robustness_emoji = {
            FormationRobustness.EXCELLENT: "🟢",
            FormationRobustness.GOOD: "🟡",
            FormationRobustness.FAIR: "🟠",
            FormationRobustness.POOR: "🔴"
        }
        
        print(f"{rank}. {robustness_emoji[profile.robustness]} {profile.name}")
        print(f"   Converges: {status_icon} | Convergence: {profile.convergence_rate*100:.0f}% "
              f"({profile.convergence_time_seconds:.2f}s)")
        print(f"   Position Error: {profile.mean_position_error_m*1000:.0f}mm | "
              f"Efficiency: {profile.energy_efficiency:.3f} J/m")
        print(f"   Contingency: Energy {profile.contingency_reserves.energy_reserve_percent*100:.0f}% | "
              f"Buffer {profile.contingency_reserves.collision_buffer_margin*100:.0f}cm")
        print()
    
    # Detailed comparison
    print("\n" + "-" * 80)
    print("DETAILED COMPARISON:\n")
    
    for profile in formations_ranked:
        print(f"\n{profile.name} Formation")
        print("-" * 40)
        print(f"  Convergence: {profile.convergence_rate*100:.0f}% in {profile.convergence_time_seconds:.2f}s")
        print(f"  Position Error: Mean {profile.mean_position_error_m*1000:.1f}mm "
              f"(Max threshold: 150mm)")
        print(f"  Energy Efficiency: {profile.energy_efficiency:.3f} J/m")
        print(f"  Min Safe Distance: {profile.min_safe_distance:.3f}m")
        print(f"  Robustness: {profile.robustness.name}")
        print(f"\n  Contingency Configuration:")
        print(f"    • Energy Reserve: {profile.contingency_reserves.energy_reserve_percent*100:.0f}% "
              f"(threshold: {4.5 * (1-profile.contingency_reserves.energy_reserve_percent):.2f}J)")
        print(f"    • Collision Buffer: {profile.contingency_reserves.collision_buffer_margin*100:.0f}cm "
              f"(threshold: {profile.contingency_reserves.min_safe_distance:.2f}m)")
        print(f"    • Position Error Threshold: {profile.contingency_reserves.max_position_error_threshold*100:.0f}cm")
        print(f"    • Convergence Timeout: {profile.contingency_reserves.convergence_timeout} steps")
        print(f"\n  Notes:")
        for line in profile.notes.strip().split('\n'):
            if line.strip():
                print(f"    {line.strip()}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS BY USE CASE:\n")
    
    recommendations = {
        "Critical/High-Reliability Missions": [
            "MANDALA (fastest, most accurate, 100% convergence)",
            "HELIX (fastest, excellent accuracy, 100% convergence)"
        ],
        "Precision-Required Tasks": [
            "FLAG (95% convergence rate, 15mm accuracy, most efficient)",
            "MANDALA (3mm accuracy - exceptional)"
        ],
        "Time-Critical Operations": [
            "MANDALA (0.69s convergence)",
            "HELIX (0.74s convergence)"
        ],
        "General/Standard Missions": [
            "CUPID (85% convergence, acceptable performance)",
            "FLAG (95% convergence rate)"
        ],
        "NOT RECOMMENDED": [
            "DRAGON (poorest performance, 6cm error, needs tuning)"
        ]
    }
    
    for use_case, formations in recommendations.items():
        print(f"📌 {use_case}:")
        for formation in formations:
            print(f"   • {formation}")
        print()
    
    print("=" * 80 + "\n")


# ============================================================================
# CONTINGENCY ADJUSTMENT FUNCTIONS
# ============================================================================

def adjust_reserves_for_degraded_operation(profile: FormationProfile, 
                                          num_failed_agents: int,
                                          total_agents: int = 20) -> ContingencyReserves:
    """
    Adjust contingency reserves when operating with failed agents.
    
    More conservative reserves needed when operating at reduced capacity.
    """
    loss_percent = num_failed_agents / total_agents
    
    adjusted_reserves = ContingencyReserves(
        energy_reserve_percent=min(
            profile.contingency_reserves.energy_reserve_percent + loss_percent * 0.1,
            0.30  # Cap at 30%
        ),
        collision_buffer_margin=min(
            profile.contingency_reserves.collision_buffer_margin + loss_percent * 0.02,
            0.10  # Cap at 10cm
        ),
        max_position_error_threshold=max(
            profile.contingency_reserves.max_position_error_threshold - loss_percent * 0.05,
            0.10  # Floor at 10cm
        ),
        convergence_timeout=int(
            profile.contingency_reserves.convergence_timeout * (1 + loss_percent)
        ),
        min_safe_distance=min(
            profile.contingency_reserves.min_safe_distance + loss_percent * 0.05,
            0.40
        )
    )
    
    return adjusted_reserves


def adjust_reserves_for_high_speed_mission(profile: FormationProfile) -> ContingencyReserves:
    """Make reserves more conservative for high-speed missions"""
    return ContingencyReserves(
        energy_reserve_percent=profile.contingency_reserves.energy_reserve_percent + 0.05,
        collision_buffer_margin=profile.contingency_reserves.collision_buffer_margin + 0.02,
        max_position_error_threshold=profile.contingency_reserves.max_position_error_threshold,
        convergence_timeout=profile.contingency_reserves.convergence_timeout,
        min_safe_distance=profile.contingency_reserves.min_safe_distance + 0.05
    )


def adjust_reserves_for_aggressive_mission(profile: FormationProfile) -> ContingencyReserves:
    """Loosen reserves for aggressive/time-critical missions (at increased risk)"""
    if profile.robustness.value > 2:  # Only for GOOD+ formations
        raise ValueError(f"Cannot use aggressive reserves for {profile.name} - insufficient robustness")
    
    return ContingencyReserves(
        energy_reserve_percent=max(
            profile.contingency_reserves.energy_reserve_percent - 0.05,
            0.10
        ),
        collision_buffer_margin=max(
            profile.contingency_reserves.collision_buffer_margin - 0.02,
            0.02
        ),
        max_position_error_threshold=profile.contingency_reserves.max_position_error_threshold * 1.2,
        convergence_timeout=int(profile.contingency_reserves.convergence_timeout * 0.8),
        min_safe_distance=profile.contingency_reserves.min_safe_distance - 0.02
    )


# ============================================================================
# EXAMPLE: SELECT RESERVES FOR YOUR MISSION
# ============================================================================

def configure_for_mission(formation_type: str, mission_type: str = 'standard',
                         num_failed_agents: int = 0) -> ContingencyReserves:
    """
    Configure contingency reserves based on formation and mission type.
    
    Args:
        formation_type: 'cupid', 'dragon', 'flag', 'helix', 'mandala'
        mission_type: 'standard', 'high_speed', 'aggressive'
        num_failed_agents: Number of non-operational agents (0-19)
    
    Returns:
        Configured ContingencyReserves for the mission
    """
    profile = FORMATION_PROFILES.get(formation_type.lower())
    if not profile:
        raise ValueError(f"Unknown formation: {formation_type}")
    
    reserves = profile.contingency_reserves
    
    # Adjust for mission type
    if mission_type.lower() == 'high_speed':
        reserves = adjust_reserves_for_high_speed_mission(profile)
    elif mission_type.lower() == 'aggressive':
        if profile.robustness == FormationRobustness.EXCELLENT:
            reserves = adjust_reserves_for_aggressive_mission(profile)
        else:
            print(f"⚠️  Warning: {formation_type} not suitable for aggressive missions")
    
    # Adjust for failed agents
    if num_failed_agents > 0:
        reserves = adjust_reserves_for_degraded_operation(profile, num_failed_agents)
    
    return reserves


# ============================================================================
# MAIN: Print all analysis
# ============================================================================

if __name__ == "__main__":
    print_formation_analysis()
    
    # Example: Get reserves for MANDALA in standard mission
    mandala_reserves = get_formation_reserves('mandala')
    print("Example: MANDALA Standard Mission")
    print(f"Energy Reserve: {mandala_reserves.energy_reserve_percent*100:.0f}%")
    print(f"Collision Buffer: {mandala_reserves.collision_buffer_margin*100:.0f}cm")
    
    # Example: Get reserves for MANDALA in high-speed mission with 2 failed agents
    mandala_hs_reserves = configure_for_mission('mandala', 'high_speed', num_failed_agents=2)
    print("\nExample: MANDALA High-Speed Mission with 2 Failed Agents")
    print(f"Energy Reserve: {mandala_hs_reserves.energy_reserve_percent*100:.0f}%")
    print(f"Collision Buffer: {mandala_hs_reserves.collision_buffer_margin*100:.0f}cm")

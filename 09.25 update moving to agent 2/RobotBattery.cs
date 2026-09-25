using UnityEngine;
using System.Collections;

/// <summary>
/// Per-robot battery. Ported from the old Robot.cs energy system, but rewired
/// so that energy consumption uses EXACTLY the same units and constants as
/// FactoryEnvManager's energy reward (moveEnergyPerTU / repairEnergyPerTU),
/// instead of a separate meter-based cost. This guarantees the battery drain
/// a robot experiences always matches the energy-cost term (EC) subtracted
/// from the global reward -- no drift between simulation and reward.
/// </summary>
[RequireComponent(typeof(AStarAgent))]
public class RobotBattery : MonoBehaviour
{
    [Header("Master Switch")]
    [Tooltip("If true, battery is ignored entirely (infinite energy). Flip off to enable real battery constraints.")]
    public bool unlimitedEnergy = true;

    [Header("Capacity")]
    [Tooltip("Total battery capacity in energy units (same units as moveEnergyPerTU / repairEnergyPerTU on FactoryEnvManager).")]
    public float maxEnergy = 100f;

    [Header("Thresholds (ignored if unlimitedEnergy == true)")]
    [Tooltip("Minimum fraction of maxEnergy required to accept a NEW job, and the threshold an idle robot " +
            "checks after finishing its current job to decide whether to go recharge. A battery drop below " +
            "this WHILE already mid-job never interrupts that job -- it always finishes first.")]
    [Range(0f, 1f)] public float lowEnergyThreshold = 0.15f;

    [Tooltip("Target fraction of maxEnergy to charge UP TO before a robot may leave the charging station. " +
            "This is NOT a dispatch-eligibility check -- a robot sitting at, say, 20% (above lowEnergyThreshold " +
            "but below this) can still be dispatched; this value only governs how full it charges once it's " +
            "actually at the station.")]
    [Range(0f, 1f)] public float minEnergyToDispatch = 0.90f;

    [Header("Charging")]
    public Transform chargingStation;
    public float rechargeRatePerSecond = 20f;

    [Header("Debug")]
    public bool debugLogs = true;

    private float currentEnergy;
    private FactoryEnvManager env;
    private Vector3 parkingOffset;

    public float CurrentEnergy => unlimitedEnergy ? maxEnergy : currentEnergy;
    public float EnergyPercent => unlimitedEnergy ? 1f : Mathf.Clamp01(currentEnergy / Mathf.Max(1f, maxEnergy));
    public bool IsLow => !unlimitedEnergy && currentEnergy < lowEnergyThreshold * maxEnergy;
    public bool CanDispatch => unlimitedEnergy || currentEnergy >= lowEnergyThreshold * maxEnergy;
    public bool IsCharging { get; private set; }

    // Same normalization constants used by FactoryEnvManager's energy reward.
    // Handy for sanity-checking that maxEnergy / minEnergyToDispatch are set sensibly
    // relative to a single worst-case job (a robot should be able to survive at
    // least one full move+repair cycle before needing a recharge).
    public float WorstCaseJobCost =>
        env != null
            ? (Mathf.Max(0, env.maxMoveTU) * Mathf.Max(0f, env.moveEnergyPerTU)) +
              (Mathf.Max(0, env.maxRepairTU) * Mathf.Max(0f, env.repairEnergyPerTU))
            : 0f;

    float MoveEnergyPerTU => env != null ? env.moveEnergyPerTU : 1f;
    float RepairEnergyPerTU => env != null ? env.repairEnergyPerTU : 1f;

    void Awake()
    {
        currentEnergy = maxEnergy;
        env = FactoryEnvManager.Instance;
    }

    void Start()
    {
        if (env == null) env = FactoryEnvManager.Instance;

        if (debugLogs && !unlimitedEnergy)
        {
            Debug.Log($"[RobotBattery] {name} init. maxEnergy={maxEnergy:F1}, worstCaseJobCost={WorstCaseJobCost:F1} " +
                      $"(maxMoveTU={env?.maxMoveTU}, maxRepairTU={env?.maxRepairTU}, moveEnergyPerTU={MoveEnergyPerTU:F2}, repairEnergyPerTU={RepairEnergyPerTU:F2}). " +
                      $"If maxEnergy < worstCaseJobCost, a robot could run out mid-job.");
        }
    }

    /// <summary>
    /// Call once, from RepairTaskManager, to space out multiple robots around
    /// their shared charging station so they don't stack visually.
    /// </summary>
    public void SetupParking(int robotIndex, float radius = 1.5f, float degreesPerRobot = 120f)
    {
        float angle = robotIndex * degreesPerRobot;
        parkingOffset = new Vector3(Mathf.Cos(angle * Mathf.Deg2Rad) * radius, 0f, Mathf.Sin(angle * Mathf.Deg2Rad) * radius);
    }

    public Vector3 GetParkingPosition()
    {
        return chargingStation != null ? chargingStation.position + parkingOffset : transform.position;
    }

    /// <summary>
    /// Drain energy for a move segment. moveTU MUST come from the same
    /// snapshot FactoryEnvManager uses for the reward (see
    /// FactoryEnvManager.TryPeekEnergyBreakdown), never recomputed independently.
    /// </summary>
    public void ConsumeMoveEnergyByTU(int moveTU)
    {
        if (unlimitedEnergy) return;
        float cost = Mathf.Max(0, moveTU) * MoveEnergyPerTU;
        Drain(cost, "move", moveTU);
    }

    /// <summary>
    /// Drain energy for a repair/inspection segment. repairTU MUST come from
    /// the same snapshot FactoryEnvManager uses for the reward.
    /// </summary>
    public void ConsumeRepairEnergyByTU(int repairTU)
    {
        if (unlimitedEnergy) return;
        float cost = Mathf.Max(0, repairTU) * RepairEnergyPerTU;
        Drain(cost, "repair", repairTU);
    }

    void Drain(float amount, string tag, float tuAmount)
    {
        float before = currentEnergy;
        currentEnergy = Mathf.Max(0f, currentEnergy - amount);
        if (debugLogs)
        {
            Debug.Log($"[RobotBattery] {name} {tag} drain: tu={tuAmount:F0} cost={amount:F3} " +
                      $"energy {before:F1} -> {currentEnergy:F1} / {maxEnergy:F1} ({EnergyPercent:P0})");
        }
    }

    /// <summary>
    /// Coroutine: recharge while parked at the charging station until the
    /// robot has enough energy to be dispatched again (or forever, if you
    /// just want it topped up -- change the while condition to '< maxEnergy').
    /// </summary>
    public IEnumerator ChargeUntilDispatchable()
    {
        IsCharging = true;
        if (debugLogs) Debug.Log($"[RobotBattery] {name} charging started at {currentEnergy:F1}/{maxEnergy:F1}.");

        while (!unlimitedEnergy && currentEnergy < minEnergyToDispatch * maxEnergy)
        {
            currentEnergy = Mathf.Min(maxEnergy, currentEnergy + rechargeRatePerSecond * Time.deltaTime);
            yield return null;
        }

        IsCharging = false;
        if (debugLogs) Debug.Log($"[RobotBattery] {name} charged to {currentEnergy:F1}/{maxEnergy:F1}, ready to dispatch.");
    }
}

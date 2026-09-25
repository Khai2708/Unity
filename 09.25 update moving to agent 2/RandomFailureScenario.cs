using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Overlapping Failure Scheduler with three cycle types:
///
/// A -> Wave burst: 5 faults simultaneously, then (after the last of those
///      5 finishes repair) 6 faults simultaneously (total 11 faults).
///      - Wave 1 (5 faults) fires at the same time.
///      - Wave 1 -> Wave 2 gap: NOT a fixed timer. Waits until ALL of wave
///        1's faults are fully repaired (IsFault == false for every one of
///        them) before firing wave 2.
///      - Wave 2 (6 faults) fires at the same time.
///      - Once wave 2's faults are all repaired, the cycle ends and (per
///        the normal scheduler) repeats the next time cycle A comes up.
///
/// B -> 2 overlapping faults (staggered)
/// C -> 1 fault (no overlap)
///
/// For cycles B and C: faults within the cycle start one after
/// another (staggered), exactly as before.
///
/// Once all faults in a cycle are repaired, the next cycle begins
/// after a configurable delay.
///
/// The cycle sequence can be fixed (a list you set in Inspector)
/// or randomly generated at runtime.
/// </summary>
public class RandomFailureScenario : MonoBehaviour
{
    // ----- Helper to get simulation time (same as before) -----
    float GetSimulationTimeSeconds()
    {
        if (factoryEnv != null)
            return factoryEnv.GetSimulationTimeSeconds();

        if (FactoryEnvManager.Instance != null)
            return FactoryEnvManager.Instance.GetSimulationTimeSeconds();

        return Time.time;
    }

    // ----- Cycle Types -----
    public enum FailureCycleType
    {
        A, // Wave burst: 5 (simultaneous) -> 6 (simultaneous) = 11 total
        B, // 2 overlapping faults
        C  // 1 fault
    }

    // ----- Wave sizes for cycle A (edit here if you want different burst sizes) -----
    // Sum of these must equal GetFaultCountForType(FailureCycleType.A).
    private static readonly int[] CycleAWaveSizes = new int[] { 5, 6 };

    // ----- Inspector References -----
    [Header("References")]
    public FactoryEnvManager factoryEnv;

    [Header("Cycle Sequence")]
    [Tooltip("If true, cycle order is randomly generated at start (length = randomSequenceLength).")]
    public bool randomCycleOrder = true;

    [Tooltip("Number of cycles to generate when using random order.")]
    public int randomSequenceLength = 20;

    [Tooltip("If randomCycleOrder is false, this exact list is used in order.")]
    public List<FailureCycleType> fixedCycleSequence = new List<FailureCycleType>();

    [Header("Timing")]
    [Tooltip("Wait before the first cycle starts.")]
    public float initialDelay = 0f;

    [Tooltip("Time between the start of consecutive faults within a B/C cycle.")]
    public float delayBetweenFaultsInCycle = 5f;

    [Tooltip("Wait after all faults in a cycle are repaired before starting the next cycle.")]
    public float delayAfterCycleCompletion = 10f;

    [Header("Cycle A Wave Timing")]
    [Tooltip("For cycle A only: the wave1->wave2 gap is NOT a fixed timer. Wave 2 waits until the number of still-faulty tunnels from wave 1 drops to this value. 0 = wait until ALL of wave 1 is fully repaired (default, matches 'after the last fault finished repair'). Higher values (e.g. 1) = wait until only that many are still being repaired.")]
    [Range(0, 5)]
    public int cycleAWaveWaitThreshold = 0;

    [Tooltip("Safety cap so the scheduler never waits forever for wave 1 to finish repairing (in case a repair stalls). 0 = no cap.")]
    public float cycleAWaveMaxWait = 120f;

    [Header("Fault Candidate Tunnels (Optional)")]
    [Tooltip("If empty, all factoryEnv.tunnels are used as candidates.")]
    public List<TunnelController> faultCandidates = new List<TunnelController>();

    [Header("Debug")]
    public bool debugLogs = true;

    // ----- Internal State -----
    private List<TunnelController> allTunnels = new List<TunnelController>();
    private Dictionary<int, TunnelController> tunnelByNodeId = new Dictionary<int, TunnelController>();

    private List<FailureCycleType> cycleQueue = new List<FailureCycleType>();
    private int currentCycleIndex = -1;

    private List<TunnelController> activeFaults = new List<TunnelController>();
    private int faultsTriggeredInCurrentCycle = 0;
    private int requiredFaultCountForCurrentCycle = 0;
    private bool cycleInProgress = false;
    private float nextCycleStartTime = 0f;

    private Coroutine cycleCoroutine;

    // ----- Unity Lifecycle -----
    void Start()
    {
        if (factoryEnv == null)
            factoryEnv = FactoryEnvManager.Instance;

        if (factoryEnv == null)
        {
            Debug.LogError("[FailureScheduler] FactoryEnvManager reference missing.");
            enabled = false;
            return;
        }

        // Initialize tunnel collections and reset their states
        allTunnels.Clear();
        tunnelByNodeId.Clear();

        if (factoryEnv.tunnels != null)
        {
            foreach (var t in factoryEnv.tunnels)
            {
                if (t == null || t.nodeId < 0) continue;

                // Disable any automatic failure/repair behaviours
                t.useItemFailure = false;
                t.autoRepairFixedDelay = false;

                // If a tunnel is already faulty, force repair it
                if (t.IsFault)
                    t.ForceRepair();

                allTunnels.Add(t);
                tunnelByNodeId[t.nodeId] = t;
            }
        }

        if (allTunnels.Count == 0)
        {
            Debug.LogError("[FailureScheduler] No valid tunnels found.");
            enabled = false;
            return;
        }

        // Build the cycle queue
        BuildCycleQueue();

        // Start the first cycle after initial delay
        nextCycleStartTime = GetSimulationTimeSeconds() + Mathf.Max(0f, initialDelay);

        if (debugLogs)
        {
            Debug.Log($"[FailureScheduler] Initialized. First cycle starts in {initialDelay:F1}s.");
            LogCycleQueue();
        }
    }

    void Update()
    {
        // If a cycle is running, do nothing here - the coroutine handles it
        if (cycleInProgress)
            return;

        // Time to start the next cycle?
        if (GetSimulationTimeSeconds() >= nextCycleStartTime)
        {
            StartNextCycle();
        }
    }

    // ----- Cycle Queue Building -----
    void BuildCycleQueue()
    {
        cycleQueue.Clear();

        if (randomCycleOrder)
        {
            for (int i = 0; i < randomSequenceLength; i++)
            {
                cycleQueue.Add((FailureCycleType)UnityEngine.Random.Range(0, 3));
            }
        }
        else
        {
            if (fixedCycleSequence == null || fixedCycleSequence.Count == 0)
            {
                Debug.LogWarning("[FailureScheduler] Fixed sequence is empty. Adding one C cycle as fallback.");
                cycleQueue.Add(FailureCycleType.C);
            }
            else
            {
                cycleQueue.AddRange(fixedCycleSequence);
            }
        }
    }

    void LogCycleQueue()
    {
        if (!debugLogs) return;

        string seq = "";
        foreach (var c in cycleQueue)
            seq += c.ToString() + " ";
        Debug.Log($"[FailureScheduler] Cycle queue: {seq}");
    }

    // ----- Starting a New Cycle -----
    void StartNextCycle()
    {
        if (cycleQueue.Count == 0)
        {
            Debug.LogWarning("[FailureScheduler] Cycle queue is empty. Nothing to run.");
            return;
        }

        currentCycleIndex = (currentCycleIndex + 1) % cycleQueue.Count;
        FailureCycleType cycleType = cycleQueue[currentCycleIndex];
        requiredFaultCountForCurrentCycle = GetFaultCountForType(cycleType);

        // Select random tunnels for this cycle (no duplicates)
        List<TunnelController> chosenTunnels = SelectRandomTunnels(requiredFaultCountForCurrentCycle);
        if (chosenTunnels.Count == 0)
        {
            Debug.LogWarning($"[FailureScheduler] No valid tunnels available for cycle {cycleType}. Skipping.");
            nextCycleStartTime = GetSimulationTimeSeconds() + 1f;
            return;
        }

        // If we couldn't get the exact number (e.g., not enough candidates), adjust required count
        requiredFaultCountForCurrentCycle = chosenTunnels.Count;

        cycleInProgress = true;
        faultsTriggeredInCurrentCycle = 0;
        activeFaults.Clear();

        if (debugLogs)
        {
            Debug.Log($"[FailureScheduler] Starting cycle {cycleType} with {requiredFaultCountForCurrentCycle} faults. " +
                      $"Cycle #{currentCycleIndex + 1}/{cycleQueue.Count}");
        }

        // Launch the coroutine that handles fault triggering and completion monitoring
        cycleCoroutine = StartCoroutine(RunCycle(cycleType, chosenTunnels));
    }

    int GetFaultCountForType(FailureCycleType type)
    {
        return type switch
        {
            FailureCycleType.A => 11, // waves: 5 + 6
            FailureCycleType.B => 2,
            FailureCycleType.C => 1,
            _ => 1
        };
    }

    List<TunnelController> SelectRandomTunnels(int count)
    {
        // Determine candidate pool
        List<TunnelController> pool = (faultCandidates != null && faultCandidates.Count > 0)
            ? new List<TunnelController>(faultCandidates)
            : new List<TunnelController>(allTunnels);

        // Remove nulls, invalid nodeIds, and already faulty tunnels (if any)
        pool.RemoveAll(t => t == null || t.nodeId < 0 || t.IsFault);

        // If we don't have enough, return what we have (or all)
        int actualCount = Mathf.Min(count, pool.Count);
        if (actualCount == 0)
            return new List<TunnelController>();

        // Shuffle and take first 'actualCount'
        for (int i = 0; i < pool.Count; i++)
        {
            int randIdx = UnityEngine.Random.Range(i, pool.Count);
            var temp = pool[i];
            pool[i] = pool[randIdx];
            pool[randIdx] = temp;
        }

        return pool.GetRange(0, actualCount);
    }

    // ----- Coroutine for Cycle Execution -----
    IEnumerator RunCycle(FailureCycleType cycleType, List<TunnelController> targets)
    {
        if (cycleType == FailureCycleType.A)
        {
            // ----- Cycle A: waves [5, 6], both simultaneous -----
            int cursor = 0;
            List<TunnelController> previousWaveTargets = null;

            for (int waveIndex = 0; waveIndex < CycleAWaveSizes.Length; waveIndex++)
            {
                int waveSize = Mathf.Min(CycleAWaveSizes[waveIndex], Mathf.Max(0, targets.Count - cursor));
                if (waveSize <= 0)
                    break;

                // ----- Gap before this wave (skip for the very first wave) -----
                if (waveIndex > 0 && previousWaveTargets != null && previousWaveTargets.Count > 0)
                {
                    // Wave 1 -> Wave 2: wait until wave 1's faults are (fully,
                    // by default) repaired - NOT a fixed timer.
                    float waitStart = GetSimulationTimeSeconds();
                    int threshold = Mathf.Clamp(cycleAWaveWaitThreshold, 0, previousWaveTargets.Count);

                    while (true)
                    {
                        int stillFaulty = 0;
                        foreach (var t in previousWaveTargets)
                        {
                            if (t != null && t.IsFault)
                                stillFaulty++;
                        }

                        if (stillFaulty <= threshold)
                        {
                            if (debugLogs)
                            {
                                Debug.Log($"[FailureScheduler] (A) Wave {waveIndex} repair check passed ({stillFaulty} still faulty, threshold {threshold}). Starting wave {waveIndex + 1}.");
                            }
                            break;
                        }

                        // Safety cap so we never wait forever if repairs stall.
                        if (cycleAWaveMaxWait > 0f && GetSimulationTimeSeconds() - waitStart >= cycleAWaveMaxWait)
                        {
                            if (debugLogs)
                            {
                                Debug.LogWarning($"[FailureScheduler] (A) Wave {waveIndex} wait exceeded cycleAWaveMaxWait ({cycleAWaveMaxWait:F1}s). Forcing wave {waveIndex + 1} to start.");
                            }
                            break;
                        }

                        yield return null;
                    }
                }

                // ----- Fire this wave: all faults activate simultaneously -----
                List<TunnelController> waveTargets = new List<TunnelController>();
                for (int i = 0; i < waveSize; i++)
                {
                    var tunnel = targets[cursor + i];
                    ActivateFault(tunnel);
                    faultsTriggeredInCurrentCycle++;
                    waveTargets.Add(tunnel);

                    if (debugLogs)
                    {
                        Debug.Log($"[FailureScheduler] (A) Wave {waveIndex + 1}: fault {i + 1}/{waveSize} triggered simultaneously on node {tunnel.nodeId} ({tunnel.name})");
                    }
                }

                cursor += waveSize;
                previousWaveTargets = waveTargets;
            }
        }
        else
        {
            // ----- Cycles B / C: original staggered behaviour -----
            for (int i = 0; i < targets.Count; i++)
            {
                // Activate the fault
                ActivateFault(targets[i]);
                faultsTriggeredInCurrentCycle++;

                if (debugLogs)
                {
                    Debug.Log($"[FailureScheduler] Fault {i + 1}/{targets.Count} triggered on node {targets[i].nodeId} ({targets[i].name})");
                }

                // Wait before triggering the next one (except after the last)
                if (i < targets.Count - 1)
                {
                    float waitStart = GetSimulationTimeSeconds();
                    while (GetSimulationTimeSeconds() < waitStart + delayBetweenFaultsInCycle)
                        yield return null;
                }
            }
        }

        // Now all faults have been triggered.
        // Wait until all are repaired.
        while (activeFaults.Count > 0)
        {
            // Clean up any null or repaired tunnels from the list
            activeFaults.RemoveAll(t => t == null || !t.IsFault);
            yield return null;
        }

        // Cycle complete
        cycleInProgress = false;
        nextCycleStartTime = GetSimulationTimeSeconds() + delayAfterCycleCompletion;

        if (debugLogs)
        {
            Debug.Log($"[FailureScheduler] Cycle {cycleType} completed. Next cycle in {delayAfterCycleCompletion:F1}s.");
        }

        cycleCoroutine = null;
    }

    // ----- Fault Activation -----
    void ActivateFault(TunnelController tunnel)
    {
        if (tunnel == null)
            return;

        // Ensure it's not already faulty
        if (tunnel.IsFault)
            tunnel.ForceRepair();

        tunnel.useItemFailure = false;
        tunnel.autoRepairFixedDelay = false;

        tunnel.ForceFail();

        activeFaults.Add(tunnel);
    }

    // ----- Public Utility (optional) -----
    /// <summary>
    /// Returns true if any cycle is currently running (faults active or waiting to trigger).
    /// </summary>
    public bool IsCycleActive => cycleInProgress;

    /// <summary>
    /// Returns the current cycle type, or null if none active.
    /// </summary>
    public FailureCycleType? CurrentCycleType => cycleInProgress ? cycleQueue[currentCycleIndex] : (FailureCycleType?)null;

    // ----- Cleanup -----
    void OnDisable()
    {
        if (cycleCoroutine != null)
            StopCoroutine(cycleCoroutine);
    }
}
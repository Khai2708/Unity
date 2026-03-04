using UnityEngine;
using System.Collections.Generic;

[DisallowMultipleComponent]
public class ProductSpawner : MonoBehaviour
{
    // ====== Graph Node ======
    [Header("Graph Node")]
    [Tooltip("그래프에서 이 Spawner의 노드 ID (0부터 시작 추천)")]
    public int nodeId = -1;
    public int NodeId => nodeId;

    [Tooltip("이 Spawner에서 바로 이어지는 첫 터널들 (갈림길 여러 개면 전부 넣기)")]
    public TunnelController[] firstTunnels;

    // ====== State ======
    // RUN       : 정상 스폰
    // HALF_HOLD : 절반 속도 스폰 (Upstream HALF_HOLD에 대응)
    // HOLD      : 스폰 완전 정지
    public enum SpawnerState { RUN, HALF_HOLD, HOLD }

    [Header("State")]
    [SerializeField] private SpawnerState state = SpawnerState.RUN;
    public bool IsRun      => state == SpawnerState.RUN;
    public bool IsHalfHold => state == SpawnerState.HALF_HOLD;
    public bool IsHold     => state == SpawnerState.HOLD;

    // ====== Visual ======
    [Header("Status Visual")]
    [Tooltip("상태 색을 바꿀 Renderer (예: 상단 판 MeshRenderer)")]
    [SerializeField] private Renderer statusRenderer;
    [SerializeField] private string colorProperty = "_Color";

    [SerializeField] private Color runColor       = new Color(0.2f, 0.8f, 0.3f, 1f);   // 초록 (정상)
    [SerializeField] private Color halfHoldColor  = new Color(1f, 0.85f, 0.1f, 1f);    // 노랑 (부분 정지)
    [SerializeField] private Color holdColor      = new Color(1f, 0.65f, 0.1f, 1f);    // 주황 (완전 HOLD)

    private Material instMat;

    // ====== Speed Adjustment ======
    [Header("Speed Adjustment")]
    [Tooltip("Base spawn interval when running at normal speed")]
    public float baseSpawnInterval = 1.5f;

    [Tooltip("Current spawn interval (affected by speed multipliers)")]
    [SerializeField] private float currentSpawnInterval = 1.5f;

    [Tooltip("Visual feedback when spawner is slowed down")]
    public Renderer speedIndicatorRenderer;
    public Color normalSpeedColor = Color.green;
    public Color slowedSpeedColor = Color.yellow;

    // Dictionary to track speed modifiers from different tunnels
    private Dictionary<TunnelController, float> speedModifiers = new Dictionary<TunnelController, float>();

    // Track the effective multiplier for debugging
    private float effectiveMultiplier = 1.0f;

    // ====== Links ======
    [Header("Links")]
    public ProductPool pool;
    public TargetPath path;
    public Transform spawnPoint;   // 없으면 path 첫 포인트 사용

    // ====== Product Settings ======
    [Header("Product Settings")]
    [Tooltip("스폰된 product가 살아 있을 최대 시간(초). 0 이하면 무제한.")]
    public float productLifetimeSeconds = 300f;  // 기본 5분

    // ====== Statistics ======
    [Header("Statistics")]
    [Tooltip("Total number of products spawned by this spawner since start/ reset.")]
    [SerializeField] private int _totalSpawnedCount;   // visible in Inspector
    public int totalSpawnedCount => _totalSpawnedCount;  // public read-only property

    // ====== Spawn Policy ======
    [Header("Spawn Policy")]
    [Tooltip("시작하자마자 자동 스폰할지 여부")]
    public bool autoStart = true;

    [Tooltip("RUN 상태일 때 기준 스폰 간격(초)")]
    public float spawnInterval = 1.5f;

    private float t;

    private void Awake()
    {
        if (statusRenderer != null)
            instMat = statusRenderer.material;
        ApplyStatusVisual();
        
        // Initialize with base interval
        currentSpawnInterval = baseSpawnInterval;
    }

    private void Update()
    {
        // 자동 스폰: HOLD가 아닌 상태에서만 시도
        if (!autoStart || pool == null || path == null || IsHold)
            return;

        // Calculate effective interval based on state and speed modifiers
        float effInterval = currentSpawnInterval;
        
        // Apply HALF_HOLD state multiplier (double interval = half speed)
        if (IsHalfHold)
            effInterval *= 2f;

        t += Time.deltaTime;
        if (t >= effInterval)
        {
            t = 0f;
            SpawnOne();
        }
    }   

    // ====== Speed Adjustment Methods ======

    /// <summary>
    /// Set a speed multiplier for this spawner from a specific tunnel
    /// </summary>
    /// <param name="multiplier">Speed multiplier (1.0 = normal, 0.5 = half speed)</param>
    /// <param name="source">Tunnel that is requesting the speed change</param>
    public void SetSpawnMultiplier(float multiplier, TunnelController source = null)
    {
        if (source != null)
        {
            // Store or update the multiplier for this tunnel
            speedModifiers[source] = Mathf.Clamp(multiplier, 0.1f, 2.0f);
        }
        
        // Calculate effective multiplier (use minimum of all modifiers)
        float minMultiplier = 1.0f;
        foreach (var mod in speedModifiers.Values)
        {
            minMultiplier = Mathf.Min(minMultiplier, mod);
        }
        
        effectiveMultiplier = minMultiplier;
        
        // Apply to spawn interval: lower multiplier = slower spawns
        currentSpawnInterval = baseSpawnInterval / effectiveMultiplier;
        
        // Update visual feedback
        UpdateSpeedVisual();
        
        // Reset timer to avoid timing issues
        t = 0f;
        
        Debug.Log($"[ProductSpawner:{name}] Speed multiplier set to {effectiveMultiplier}x (interval: {currentSpawnInterval:F2}s)");
    }

    /// <summary>
    /// Clear all speed modifiers (use when resetting the system)
    /// </summary>
    public void ClearAllSpeedModifiers()
    {
        speedModifiers.Clear();
        effectiveMultiplier = 1.0f;
        currentSpawnInterval = baseSpawnInterval;
        UpdateSpeedVisual();
        t = 0f;
    }

    /// <summary>
    /// Remove a speed multiplier from a specific tunnel
    /// </summary>
    public void ClearSpawnMultiplier(TunnelController source)
    {
        if (source != null && speedModifiers.ContainsKey(source))
        {
            speedModifiers.Remove(source);
            
            // Recalculate without this modifier
            float minMultiplier = 1.0f;
            foreach (var mod in speedModifiers.Values)
            {
                minMultiplier = Mathf.Min(minMultiplier, mod);
            }
            
            effectiveMultiplier = minMultiplier;
            currentSpawnInterval = baseSpawnInterval / effectiveMultiplier;
            
            // Update visual feedback
            UpdateSpeedVisual();
            
            // Reset timer to avoid timing issues
            t = 0f;
            
            Debug.Log($"[ProductSpawner:{name}] Speed multiplier removed. Now {effectiveMultiplier}x (interval: {currentSpawnInterval:F2}s)");
        }
    }

    /// <summary>
    /// Get current spawn speed multiplier
    /// </summary>
    public float GetSpawnMultiplier()
    {
        return effectiveMultiplier;
    }

    /// <summary>
    /// Update visual indicator based on current speed
    /// </summary>
    private void UpdateSpeedVisual()
    {
        if (speedIndicatorRenderer != null)
        {
            Material mat = speedIndicatorRenderer.material;
            if (effectiveMultiplier < 0.8f)
                mat.color = slowedSpeedColor;
            else
                mat.color = normalSpeedColor;
        }
    }

    /// <summary>
    /// Get current spawn interval (for debugging)
    /// </summary>
    public float GetCurrentSpawnInterval()
    {
        return currentSpawnInterval;
    }

    /// <summary>
    /// 수동 스폰 호출.
    /// </summary>
    public void SpawnOne()
    {
        // 완전 HOLD면 스폰 금지
        if (IsHold || pool == null || path == null)
            return;

        var go = pool.Get();
        if (go == null)
            return;

        Vector3 pos = spawnPoint ? spawnPoint.position : path.GetPoint(0).position;
        go.transform.SetPositionAndRotation(pos, Quaternion.identity);

        var follower = go.GetComponent<PathFollower>();
        if (!follower) follower = go.AddComponent<PathFollower>();
        follower.SetPath(path);

        var ret = go.GetComponent<ReturnToPoolOnFinish>();
        if (!ret) ret = go.AddComponent<ReturnToPoolOnFinish>();
        ret.pool = pool;

        if (productLifetimeSeconds > 0f)
            ret.lifetimeSeconds = productLifetimeSeconds;

        // Increment spawn counter
        _totalSpawnedCount++;
    }

    // ====== External control (from upstream tunnel) ======

    public void SetState(SpawnerState newState)
    {
        if (state == newState) return;
        state = newState;
        ApplyStatusVisual();
        
        // Reset timer on state change to avoid timing issues
        t = 0f;
    }

    public void EnterHold()
    {
        if (state == SpawnerState.HOLD) return;
        state = SpawnerState.HOLD;
        ApplyStatusVisual();
    }

    public void EnterRun()
    {
        if (state == SpawnerState.RUN) return;
        state = SpawnerState.RUN;
        ApplyStatusVisual();
    }

    public void EnterHalfHold()
    {
        if (state == SpawnerState.HALF_HOLD) return;
        state = SpawnerState.HALF_HOLD;
        ApplyStatusVisual();
    }

    public void OnUpstreamHold()      => EnterHold();
    public void OnUpstreamResume()    => EnterRun();
    public void OnUpstreamHalfHold()  => EnterHalfHold();

    // ====== Visual ======
    private void ApplyStatusVisual()
    {
        if (instMat == null) return;

        var prop = string.IsNullOrEmpty(colorProperty) ? "_Color" : colorProperty;
        Color c = runColor;

        switch (state)
        {
            case SpawnerState.RUN:       c = runColor;      break;
            case SpawnerState.HALF_HOLD: c = halfHoldColor; break;
            case SpawnerState.HOLD:      c = holdColor;     break;
        }

        if (instMat.HasProperty(prop))
            instMat.SetColor(prop, c);
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        Color c = runColor;
        switch (state)
        {
            case SpawnerState.RUN:       c = runColor;      break;
            case SpawnerState.HALF_HOLD: c = halfHoldColor; break;
            case SpawnerState.HOLD:      c = holdColor;     break;
        }

        Gizmos.color = c;
        Gizmos.DrawWireCube(
            transform.position + Vector3.up * 0.1f,
            new Vector3(0.18f, 0.02f, 0.18f)
        );
    }
#endif
}
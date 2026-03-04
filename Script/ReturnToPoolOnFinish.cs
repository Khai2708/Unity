using UnityEngine;

[DisallowMultipleComponent]
public class ReturnToPoolOnFinish : MonoBehaviour
{
    [Header("Pool 설정")]
    public ProductPool pool;

    [Header("수명 설정")]
    [Tooltip("0 이하면 시간 기반 회수는 안 하고, 경로 완료 시에만 회수")]
    public float lifetimeSeconds = 300f;

    private PathFollower follower;
    private float spawnTime;
    private bool subscribed = false;

    private void Awake()
    {
        follower = GetComponent<PathFollower>();
    }

    private void OnEnable()
    {
        spawnTime = Time.time;

        // PathFollower의 경로 완료 이벤트에 구독
        if (follower != null && !subscribed)
        {
            follower.OnFinished += HandlePathFinished;
            subscribed = true;
        }
    }

    private void OnDisable()
    {
        if (follower != null && subscribed)
        {
            follower.OnFinished -= HandlePathFinished;
            subscribed = false;
        }
    }

    private void Update()
    {
        // 수명 초과로 회수할 때는 "생산 완료"로 보지 않음
        if (lifetimeSeconds > 0f &&
            Time.time - spawnTime >= lifetimeSeconds)
        {
            DoReturn(countAsThroughput: false);
        }
    }

    private void HandlePathFinished()
    {
        // 경로 끝까지 간 경우만 생산량(throughput)으로 카운트
        DoReturn(countAsThroughput: true);
    }

    private void DoReturn(bool countAsThroughput)
    {
        if (!gameObject.activeInHierarchy)
            return;

        // 🔹 여기서 전역 throughput 카운터 증가
        if (countAsThroughput && FactoryEnvManager.Instance != null)
        {
            FactoryEnvManager.Instance.RegisterProductExit();
        }

        if (pool != null)
        {
            pool.Return(gameObject);
        }
        else
        {
            gameObject.SetActive(false);
        }
    }
}

using UnityEngine;
using System;

[RequireComponent(typeof(Transform))]
public class PathFollower : MonoBehaviour
{
    [Header("Movement")]
    public float speed = 1.2f;
    public float arriveEps = 0.02f;

    [Header("Anti-overlap (segment gating)")]
    [Tooltip("다음 포인트를 다른 제품이 선점하면 그 포인트에 예약이 걸릴 때까지 대기")]
    public bool useWaypointReservation = true;

    [Tooltip("다음 포인트 직전에서 멈출 여유 거리(간섭 방지용)")]
    public float holdGap = 0.0f;

    private TargetPath path;
    private int   segIdx;         // i -> i+1
    private float segT;           // 0..1
    private bool  finished;
    private bool  paused;

    // 경로 갈아타기 시 이번 프레임에서 Advance를 한 번 건너뛰기 위한 플래그
    private bool  skipAdvanceOnce;

    // 현재/다음 포인트의 Reservable
    private Reservable curRes;
    private Reservable nextRes;

    // 이벤트
    public Action OnFinished;
    public Action<Transform, WaypointMarker> OnReachedPoint; // i+1 포인트 도착 알림

    // ==== 외부 노출(디버그용) ====
    public Transform CurrentSegmentEnd  => (path != null && segIdx < path.Count - 1) ? path.GetPoint(segIdx + 1) : null;
    public int CurrentSegmentIndex      => segIdx;

    // 외부에서 읽을 수 있는 일시정지 상태
    public bool IsPaused => paused;

    // ====== API ======
    public void SetPath(TargetPath newPath)
    {
        path = newPath;
        finished = false;
        paused   = false;

        if (path == null || path.Count < 2)
        {
            enabled = false;
            return;
        }

        segIdx = 0;
        segT   = 0f;
        transform.SetPositionAndRotation(path.GetPoint(0).position, Quaternion.identity);
        enabled = true;

        // 시작 포인트 예약
        if (useWaypointReservation)
        {
            TryReserveCurrent();
            TryReserveNext();
        }
    }

    /// <summary>
    /// 다른 TargetPath로 갈아타기 (분기/다리 진입 등에서 사용)
    /// </summary>
    public void SwitchPath(TargetPath newPath, int startIndex = 0, bool snapToPoint = true)
    {
        // 기존 경로의 예약 모두 해제
        ReleaseAllReservations();

        path     = newPath;
        finished = false;
        paused   = false;

        if (path == null || path.Count < 2)
        {
            enabled = false;
            return;
        }

        segIdx = Mathf.Clamp(startIndex, 0, path.Count - 1);
        segT   = 0f;

        if (snapToPoint)
        {
            var p = path.GetPoint(segIdx);
            if (p != null)
            {
                transform.position = p.position;
            }
        }

        curRes  = null;
        nextRes = null;

        if (useWaypointReservation)
        {
            // 새 경로 시작 포인트 및 다음 포인트 예약
            TryReserveCurrent();
            TryReserveNext();
        }

        // 이번 프레임에는 이전 경로에 대한 Advance를 건너뛰기
        skipAdvanceOnce = true;
        finished        = false;
        enabled         = true;
    }

    public void Pause()  => paused = true;
    public void Resume() => paused = false;

    /// <summary>
    /// (예: 터널에서 큐로 텔레포트하기 직전) 현재/다음 예약 해제용
    /// </summary>
    public void ReleaseCurrentReservation()
    {
        if (curRes != null)
        {
            curRes.Release(this);
            curRes = null;
        }
    }

    public void ReleaseNextReservation()
    {
        if (nextRes != null)
        {
            nextRes.Release(this);
            nextRes = null;
        }
    }

    public void ReleaseAllReservations()
    {
        ReleaseNextReservation();
        ReleaseCurrentReservation();
    }

    private void OnDisable()
    {
        // 안전장치 — 어디서 비활성화되더라도 락 누수 방지
        ReleaseAllReservations();
    }

    // ====== 내부 구현 ======
    private void Update()
    {
        if (finished || paused || path == null || !enabled) return;

        // 세그먼트 끝에 가까워지면, 다음 포인트 예약 안 됐으면 예약 시도
        if (useWaypointReservation && nextRes == null)
            TryReserveNext();

        // 다음 포인트가 다른 제품에게 선점되어 있고(=nextRes==null),
        // 우리는 아직 예약을 못했으면 끝점 근처에서 멈춰 기다림
        if (useWaypointReservation && nextRes == null)
        {
            var a = path.GetPoint(segIdx).position;
            var b = path.GetPoint(segIdx + 1).position;

            var p = Vector3.Lerp(a, b, segT);
            float remain = Vector3.Distance(p, b);
            if (remain <= holdGap)
                return; // 대기
        }

        // 일반 이동
        StepMove();
    }

    private void StepMove()
    {
        if (segIdx >= path.Count - 1)
        {
            Finish();
            return;
        }

        var a = path.GetPoint(segIdx).position;
        var b = path.GetPoint(segIdx + 1).position;

        var ab  = b - a;
        var len = ab.magnitude;
        if (len < 1e-6f)
        {
            ReachPoint(segIdx + 1);
            Advance();
            return;
        }

        float dt = Mathf.Clamp01((speed * Time.deltaTime) / Mathf.Max(len, 1e-6f));
        segT = Mathf.Min(1f, segT + dt);
        transform.position = Vector3.Lerp(a, b, segT);

        var dir = b - a;
        dir.y = 0f;
        if (dir.sqrMagnitude > 1e-6f)
            transform.rotation = Quaternion.LookRotation(dir);

        if (segT >= 1f - arriveEps)
        {
            // 끝점 도달 — 반드시 next 예약이 있거나(겹침 방지) 예약을 사용하지 않을 때만 진입
            if (!useWaypointReservation || nextRes != null)
            {
                ReachPoint(segIdx + 1);

                // ReachPoint 안에서 JunctionPoint에 의해 SwitchPath가 호출되면
                // 이전 경로에 대한 Advance는 한 번 건너뛴다.
                if (skipAdvanceOnce)
                {
                    skipAdvanceOnce = false;
                    return;
                }

                Advance();
            }
        }
    }

    private void ReachPoint(int pointIndex)
    {
        var t = path.GetPoint(pointIndex);
        if (!t) return;

        var mk = t.GetComponent<WaypointMarker>();
        OnReachedPoint?.Invoke(t, mk);

        // 분기점(JunctionPoint)이 있을 경우, 여기서 다른 경로로 갈아타기를 시도
        var jp = t.GetComponent<JunctionPoint>();
        if (jp != null)
        {
            jp.TryRedirect(this, path, pointIndex);
        }
    }

    private void Advance()
    {
        // 다음 포인트로 승격
        if (useWaypointReservation)
        {
            // 현재 락 해제
            ReleaseCurrentReservation();

            // next -> current 로 승격
            curRes  = nextRes;
            nextRes = null;
        }

        segIdx++;
        segT = 0f;

        if (segIdx >= path.Count - 1)
        {
            Finish();
        }
        else if (useWaypointReservation)
        {
            // 새 세그먼트가 시작되었으니 다음 포인트 예약을 미리 시도
            TryReserveNext();
        }
    }

    private void Finish()
    {
        if (finished) return;
        finished = true;
        enabled  = false;
        ReleaseAllReservations();
        OnFinished?.Invoke();
    }

    private void TryReserveCurrent()
    {
        var t = path.GetPoint(0);
        if (!t) return;

        var r = t.GetComponent<Reservable>();
        if (r == null) return;

        if (r.TryReserve(this))
        {
            curRes = r;
        }
        else
        {
            // 예약 실패 시에는 시작 지점에 대한 락만 못 잡는 상태.
            // 이동 로직 자체는 유지.
        }
    }

    private void TryReserveNext()
    {
        if (segIdx >= path.Count - 1) return;

        var t = path.GetPoint(segIdx + 1);
        if (!t) return;

        var r = t.GetComponent<Reservable>();
        if (r == null) return;

        if (r.TryReserve(this))
        {
            nextRes = r;
        }
        else
        {
            // 예약 실패 — Update에서 holdGap 이내에서 대기하게 됨
        }
    }
}

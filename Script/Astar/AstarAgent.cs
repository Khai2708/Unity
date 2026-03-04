using System.Collections.Generic;
using UnityEngine;
using System;

[RequireComponent(typeof(Rigidbody))]
public class AStarAgent : MonoBehaviour
{
    public GridManager grid;
    public Transform target;
    public float moveSpeed = 3f;
    public float stoppingDistance = 0.1f;

    Rigidbody rb;

    List<Vector3> path;
    int pathIndex = 0;
    bool hasPath = false;

    public Action OnPathFinished;
    public bool IsMoving => hasPath;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        rb.interpolation = RigidbodyInterpolation.Interpolate;
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
            RecalculatePath();
    }

    void FixedUpdate()
    {
        if (hasPath)
            FollowPath();
    }

    public void RecalculatePath()
    {
        if (grid == null || target == null)
        {
            hasPath = false;
            path = null;
            pathIndex = 0;
            return;
        }

        path = grid.GetPath(transform.position, target.position);
        if (path == null || path.Count == 0)
        {
            hasPath = false;
            pathIndex = 0;
            return;
        }

        pathIndex = 0;
        hasPath = true;
    }

    public void SetTarget(Transform newTarget, bool recalcPath = true)
    {
        if (newTarget == null)
            return;

        target = newTarget;

        if (recalcPath)
            RecalculatePath();
    }

    public void ClearPath()
    {
        hasPath = false;
        path = null;
        pathIndex = 0;
    }

    void FollowPath()
    {
        if (!hasPath || path == null || pathIndex >= path.Count)
            return;

        Vector3 nodePos = path[pathIndex];
        nodePos.y = transform.position.y;

        Vector3 toNode = nodePos - transform.position;
        float distToNode = toNode.magnitude;

        float step = moveSpeed * Time.fixedDeltaTime;

        float arriveThreshold = Mathf.Max(stoppingDistance, step * 0.5f);
        if (distToNode <= arriveThreshold)
        {
            rb.MovePosition(nodePos);
            pathIndex++;

            if (pathIndex >= path.Count)
            {
                hasPath = false;

                float distToTarget = target != null
                    ? Vector3.Distance(transform.position, new Vector3(target.position.x, transform.position.y, target.position.z))
                    : 0f;

                if (target != null && distToTarget > stoppingDistance * 2f)
                {
                    RecalculatePath();

                    if (!hasPath)
                        OnPathFinished?.Invoke();
                }
                else
                {
                    OnPathFinished?.Invoke();
                }
            }

            return;
        }

        Vector3 dir = toNode / distToNode;
        float moveDist = Mathf.Min(step, distToNode);
        Vector3 newPos = transform.position + dir * moveDist;
        rb.MovePosition(newPos);

        if (dir != Vector3.zero)
        {
            Quaternion targetRot = Quaternion.LookRotation(dir);
            transform.rotation = Quaternion.RotateTowards(
                transform.rotation,
                targetRot,
                7200f * Time.fixedDeltaTime);
        }
    }
}

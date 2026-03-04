using System.Collections.Generic;
using UnityEngine;

public class GridManager : MonoBehaviour
{
    [Header("Grid Settings")]
    public Vector2 gridWorldSize = new Vector2(20f, 20f);

    [Tooltip("기준 노드 반지름 (실제 그리드는 이 값의 절반 크기로 생성되어 2배 촘촘해집니다)")]
    public float nodeRadius = 0.4f;

    [Header("Robot Physical Size")]
    [Tooltip("로봇의 실제 반지름")]
    public float robotRadius = 0.5f;

    [Tooltip("장애물 체크 여유 배율 (1.5 ~ 2.0 추천)")]
    public float safetyMultiplier = 1.8f;

    public LayerMask obstacleMask;

    Node[,] grid;

    // 실제 사용되는 노드 반지름/지름 (nodeRadius의 절반 -> 2배 촘촘)
    float actualNodeRadius;
    float nodeDiameter;
    int gridSizeX, gridSizeY;

    [HideInInspector] public List<Node> debugPath;

    // === 디버그 그리드/경로 표시 토글 ===
    [Header("Debug View")]
    [Tooltip("씬/게임 뷰에 그리드 & 경로 Gizmo를 표시할지 여부")]
    public bool showDebugGizmos = true;

    [Tooltip("플레이 중 이 키를 눌러 디버그 뷰를 토글 (예: G)")]
    public KeyCode toggleDebugKey = KeyCode.G;

    void Awake()
    {
        // === 여기서 2배 촘촘하게 만듦 ===
        // 인스펙터에서 설정한 nodeRadius의 절반을 실제 반지름으로 사용
        actualNodeRadius = nodeRadius * 0.5f;

        nodeDiameter = actualNodeRadius * 2f;
        gridSizeX = Mathf.RoundToInt(gridWorldSize.x / nodeDiameter);
        gridSizeY = Mathf.RoundToInt(gridWorldSize.y / nodeDiameter);
        CreateGrid();
    }

    void Update()
    {
        // 플레이 중 키로 토글
        if (Input.GetKeyDown(toggleDebugKey))
        {
            showDebugGizmos = !showDebugGizmos;
        }
    }

    /// <summary>
    /// UI 버튼에서 호출할 수 있는 디버그 토글 함수
    /// (Button OnClick에 GridManager를 드래그한 후 이 함수 선택)
    /// </summary>
    public void ToggleDebugGizmos()
    {
        showDebugGizmos = !showDebugGizmos;
    }

    void CreateGrid()
    {
        grid = new Node[gridSizeX, gridSizeY];

        Vector3 worldBottomLeft =
            transform.position
            - Vector3.right * gridWorldSize.x / 2f
            - Vector3.forward * gridWorldSize.y / 2f;

        float checkRadius = robotRadius * safetyMultiplier;

        for (int x = 0; x < gridSizeX; x++)
        for (int y = 0; y < gridSizeY; y++)
        {
            Vector3 worldPoint =
                worldBottomLeft
                + Vector3.right * (x * nodeDiameter + actualNodeRadius)
                + Vector3.forward * (y * nodeDiameter + actualNodeRadius);

            bool walkable =
                !Physics.CheckSphere(worldPoint, checkRadius, obstacleMask);

            grid[x, y] = new Node(walkable, worldPoint, x, y);
        }
    }

    public Node NodeFromWorldPoint(Vector3 worldPos)
    {
        float percentX =
            (worldPos.x - (transform.position.x - gridWorldSize.x / 2f)) / gridWorldSize.x;
        float percentY =
            (worldPos.z - (transform.position.z - gridWorldSize.y / 2f)) / gridWorldSize.y;

        percentX = Mathf.Clamp01(percentX);
        percentY = Mathf.Clamp01(percentY);

        int x = Mathf.RoundToInt((gridSizeX - 1) * percentX);
        int y = Mathf.RoundToInt((gridSizeY - 1) * percentY);

        return grid[x, y];
    }

    public List<Node> GetNeighbours(Node node)
    {
        List<Node> neighbours = new List<Node>();

        int[,] dirs = new int[,]
        {
            { 0,  1 },
            { 1,  0 },
            { 0, -1 },
            { -1, 0 }
        };

        for (int i = 0; i < 4; i++)
        {
            int checkX = node.gridX + dirs[i, 0];
            int checkY = node.gridY + dirs[i, 1];

            if (checkX >= 0 && checkX < gridSizeX &&
                checkY >= 0 && checkY < gridSizeY)
            {
                neighbours.Add(grid[checkX, checkY]);
            }
        }

        return neighbours;
    }

    int GetDistance(Node a, Node b)
    {
        int dstX = Mathf.Abs(a.gridX - b.gridX);
        int dstY = Mathf.Abs(a.gridY - b.gridY);
        return 10 * (dstX + dstY);
    }

    public List<Vector3> GetPath(Vector3 startPos, Vector3 targetPos)
    {
        Node startNode = NodeFromWorldPoint(startPos);
        Node targetNode = NodeFromWorldPoint(targetPos);

        foreach (Node n in grid)
        {
            n.gCost = int.MaxValue;
            n.hCost = 0;
            n.parent = null;
        }

        startNode.gCost = 0;
        startNode.hCost = GetDistance(startNode, targetNode);

        List<Node> openSet = new List<Node>();
        HashSet<Node> closedSet = new HashSet<Node>();
        openSet.Add(startNode);

        while (openSet.Count > 0)
        {
            Node currentNode = openSet[0];
            for (int i = 1; i < openSet.Count; i++)
            {
                if (openSet[i].fCost < currentNode.fCost ||
                    (openSet[i].fCost == currentNode.fCost &&
                     openSet[i].hCost < currentNode.hCost))
                {
                    currentNode = openSet[i];
                }
            }

            openSet.Remove(currentNode);
            closedSet.Add(currentNode);

            if (currentNode == targetNode)
                return RetracePath(startNode, targetNode);

            foreach (Node neighbour in GetNeighbours(currentNode))
            {
                if (!neighbour.walkable || closedSet.Contains(neighbour))
                    continue;

                int newCost =
                    currentNode.gCost + GetDistance(currentNode, neighbour);

                if (newCost < neighbour.gCost)
                {
                    neighbour.gCost = newCost;
                    neighbour.hCost = GetDistance(neighbour, targetNode);
                    neighbour.parent = currentNode;

                    if (!openSet.Contains(neighbour))
                        openSet.Add(neighbour);
                }
            }
        }

        debugPath = null;
        return null;
    }

    List<Vector3> RetracePath(Node startNode, Node endNode)
    {
        List<Node> pathNodes = new List<Node>();
        Node currentNode = endNode;

        while (currentNode != startNode)
        {
            pathNodes.Add(currentNode);
            currentNode = currentNode.parent;
        }
        pathNodes.Reverse();

        debugPath = pathNodes;

        List<Vector3> path = new List<Vector3>();
        foreach (Node n in pathNodes)
            path.Add(n.worldPosition);

        return path;
    }

#if UNITY_EDITOR
    void OnDrawGizmos()
    {
        // 🔹 디버그 뷰 꺼져 있으면 아무것도 그리지 않음
        if (!showDebugGizmos)
            return;

        Gizmos.color = Color.gray;
        Gizmos.DrawWireCube(
            transform.position,
            new Vector3(gridWorldSize.x, 1f, gridWorldSize.y)
        );

        if (grid != null)
        {
            foreach (Node n in grid)
            {
                Gizmos.color = n.walkable ? Color.white : Color.red;
                Gizmos.DrawCube(
                    n.worldPosition,
                    Vector3.one * (nodeDiameter * 0.9f)
                );
            }
        }

        if (debugPath != null)
        {
            Gizmos.color = Color.cyan;
            foreach (Node n in debugPath)
            {
                Gizmos.DrawCube(
                    n.worldPosition,
                    Vector3.one * (nodeDiameter * 0.9f)
                );
            }
        }
    }
#endif
}

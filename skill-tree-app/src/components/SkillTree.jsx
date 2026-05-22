import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  useReactFlow,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';
import dagre from 'dagre';

import SkillNode from './SkillNode';
import { parseMermaidToTree } from '../utils/mermaidParser';
import { mermaidText } from '../data/skillTree';

// Custom node types
const nodeTypes = {
  skillNode: SkillNode,
};

// Helper to generate unique ID
let idCounter = 0;
const getId = () => `node_${idCounter++}`;

// Convert tree to initial nodes and edges with expand/collapse state
function treeToFlowElements(tree, parentId = null, depth = 0, expandedNodes = new Set()) {
  const nodes = [];
  const edges = [];

  // If parent is collapsed, don't render this node (except root)
  if (parentId && !expandedNodes.has(parentId)) {
    return { nodes, edges };
  }

  const nodeId = tree.id || getId();
  // Store expanded state; root and nodes with children are expanded by default
  const isExpanded = tree.children.length > 0 ? (expandedNodes.has(nodeId) ? expandedNodes.get(nodeId) : true) : false;
  
  nodes.push({
    id: nodeId,
    type: 'skillNode',
    data: {
      label: tree.label,
      children: tree.children,
      isExpanded,
      onToggle: () => {}, // placeholder, will be set later
      onEdit: () => {},
      onDelete: () => {},
      onViewDetails: () => {},
    },
    position: { x: 0, y: depth * 200 }, // placeholder, will be updated by layout
    style: {
      width: 250,
      minHeight: 80,
    },
  });

  if (parentId) {
    edges.push({
      id: `e${parentId}-${nodeId}`,
      source: parentId,
      target: nodeId,
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#555' },
    });
  }

  // Recursively process children only if expanded
  if (tree.children.length > 0 && isExpanded) {
    tree.children.forEach(child => {
      const childElements = treeToFlowElements(child, nodeId, depth + 1, expandedNodes);
      nodes.push(...childElements.nodes);
      edges.push(...childElements.edges);
    });
  }

  return { nodes, edges };
}

// Layout algorithm: simple tree layout
function applyTreeLayout(nodes, edges, direction = 'TB') {
  // Build adjacency list
  const adjacency = {};
  nodes.forEach(node => {
    adjacency[node.id] = [];
  });
  edges.forEach(edge => {
    adjacency[edge.source].push(edge.target);
  });

  // Find root (node with no incoming edges)
  const targets = new Set(edges.map(e => e.target));
  let rootId = nodes.find(n => !targets.has(n.id))?.id;
  if (!rootId) {
    // fallback: use first node
    rootId = nodes[0]?.id;
  }

  // DFS to assign levels and positions
  const levels = {};
  const visited = new Set();
  const nodeWidth = 250;
  const nodeHeight = 100;
  const levelSpacing = 150;
  const siblingSpacing = 50;

  function dfs(nodeId, level) {
    if (visited.has(nodeId)) return;
    visited.add(nodeId);
    if (!levels[level]) levels[level] = [];
    levels[level].push(nodeId);
    adjacency[nodeId].forEach(childId => {
      dfs(childId, level + 1);
    });
  }

  dfs(rootId, 0);

  // Assign positions
  const nodeMap = {};
  nodes.forEach(node => {
    nodeMap[node.id] = node;
  });

  Object.keys(levels).forEach(level => {
    const nodesInLevel = levels[level];
    const totalWidth = nodesInLevel.length * (nodeWidth + siblingSpacing);
    nodesInLevel.forEach((nodeId, index) => {
      const node = nodeMap[nodeId];
      if (direction === 'TB') {
        node.position = {
          x: index * (nodeWidth + siblingSpacing) - totalWidth / 2,
          y: level * (nodeHeight + levelSpacing),
        };
      } else {
        node.position = {
          x: level * (nodeWidth + levelSpacing),
          y: index * (nodeHeight + siblingSpacing) - totalWidth / 2,
        };
      }
    });
  });

  return nodes;
}

// Dagre layout algorithm
function layoutWithDagre(nodes, edges, direction = 'TB') {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: direction, nodesep: 50, ranksep: 100 });
  g.setDefaultEdgeLabel(() => ({}));

  // Add nodes to dagre graph
  nodes.forEach(node => {
    g.setNode(node.id, { width: 250, height: 100 });
  });

  // Add edges
  edges.forEach(edge => {
    g.setEdge(edge.source, edge.target);
  });

  // Calculate layout
  dagre.layout(g);

  // Get positions and assign to nodes
  const nodeMap = {};
  nodes.forEach(node => {
    nodeMap[node.id] = node;
  });

  g.nodes().forEach(nodeId => {
    const node = nodeMap[nodeId];
    if (node) {
      const pos = g.node(nodeId);
      node.position = { x: pos.x, y: pos.y };
    }
  });

  return nodes;
}

const SkillTree = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [expandedNodes, setExpandedNodes] = useState(new Map()); // nodeId -> boolean
  const [selectedNode, setSelectedNode] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState('view'); // 'view', 'edit', 'add'
  const [treeData, setTreeData] = useState(null);
  const { fitView } = useReactFlow();

  // Parse mermaid text into tree
  useEffect(() => {
    try {
      const tree = parseMermaidToTree(mermaidText);
      setTreeData(tree);
      // Initialize expanded nodes: all nodes with children are expanded
      const expanded = new Map();
      const traverse = (node) => {
        if (node.children.length > 0) {
          expanded.set(node.id, true);
          node.children.forEach(traverse);
        }
      };
      traverse(tree);
      setExpandedNodes(expanded);
    } catch (error) {
      console.error('Failed to parse mermaid:', error);
    }
  }, []);

  // Convert tree to flow elements when treeData or expandedNodes change
  useEffect(() => {
    if (!treeData) return;
    const { nodes: newNodes, edges: newEdges } = treeToFlowElements(treeData, null, 0, expandedNodes);
    // Apply layout
    const laidOutNodes = layoutWithDagre(newNodes, newEdges);
    setNodes(laidOutNodes);
    setEdges(newEdges);
    // Fit view after layout
    setTimeout(() => fitView(), 100);
  }, [treeData, expandedNodes, setNodes, setEdges, fitView]);

  // Toggle node expand/collapse
  const toggleNode = useCallback((nodeId) => {
    setExpandedNodes(prev => {
      const next = new Map(prev);
      next.set(nodeId, !next.get(nodeId));
      return next;
    });
  }, []);

  // Update node data with callbacks
  useEffect(() => {
    setNodes(nodes =>
      nodes.map(node => ({
        ...node,
        data: {
          ...node.data,
          onToggle: () => toggleNode(node.id),
          onEdit: () => {
            setSelectedNode(node);
            setModalMode('edit');
            setIsModalOpen(true);
          },
          onDelete: () => {
            // For simplicity, we'll just remove the node from the tree
            // In a real app, you'd need to update the treeData structure
            alert('删除功能需要在树数据结构中实现');
          },
          onViewDetails: () => {
            setSelectedNode(node);
            setModalMode('view');
            setIsModalOpen(true);
          },
        },
      }))
    );
  }, [toggleNode, setNodes]);

  // Add new node
  const addNode = useCallback((parentId) => {
    const newNode = {
      id: getId(),
      type: 'skillNode',
      data: {
        label: '新节点',
        children: [],
        isExpanded: false,
        onToggle: () => {},
        onEdit: () => {},
        onDelete: () => {},
        onViewDetails: () => {},
      },
      position: { x: 0, y: 0 },
    };
    setNodes(nds => addNodeToTree(nds, parentId, newNode));
    // Also need to update edges
    if (parentId) {
      setEdges(eds => addEdge({
        id: `e${parentId}-${newNode.id}`,
        source: parentId,
        target: newNode.id,
        type: 'smoothstep',
        animated: true,
        style: { stroke: '#555' },
      }, eds));
    }
  }, [setNodes, setEdges]);

  // Helper to add node to tree (simplified)
  function addNodeToTree(nodes, parentId, newNode) {
    // For simplicity, we'll just add to the end and let layout handle positions
    // In a real app, you'd need to update the tree structure
    return [...nodes, newNode];
  }

  // Handle node click to select
  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node);
  }, []);

  // Handle modal close
  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedNode(null);
  };

  // Handle save from modal
  const handleSave = (updatedData) => {
    if (selectedNode) {
      setNodes(nds =>
        nds.map(n =>
          n.id === selectedNode.id
            ? { ...n, data: { ...n.data, ...updatedData } }
            : n
        )
      );
    }
    closeModal();
  };

  // Add root child
  const addRootChild = () => {
    if (treeData) {
      addNode(treeData.id);
    }
  };

  return (
    <div className="skill-tree-container">
      <div className="toolbar">
        <button onClick={addRootChild}>添加子节点</button>
        <button onClick={() => fitView()}>适应视图</button>
        <button onClick={() => {
          // Toggle all nodes
          const allExpanded = new Map();
          const traverse = (node) => {
            allExpanded.set(node.id, true);
            node.children.forEach(traverse);
          };
          if (treeData) traverse(treeData);
          setExpandedNodes(allExpanded);
        }}>展开全部</button>
        <button onClick={() => setExpandedNodes(new Map())}>折叠全部</button>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
        style={{ background: '#f7fafc' }}
      >
        <Background color="#aaa" gap={16} />
        <Controls />
        <MiniMap />
      </ReactFlow>

      {/* Modal for view/edit/add */}
      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>{modalMode === 'view' ? '查看详情' : modalMode === 'edit' ? '编辑节点' : '添加节点'}</h2>
            <div className="modal-content">
              {modalMode === 'view' ? (
                <div>
                  <p><strong>标签:</strong> {selectedNode?.data.label}</p>
                  <p><strong>子节点数:</strong> {selectedNode?.data.children?.length || 0}</p>
                </div>
              ) : (
                <form onSubmit={(e) => {
                  e.preventDefault();
                  const formData = new FormData(e.target);
                  handleSave({ label: formData.get('label') });
                }}>
                  <label>
                    标签:
                    <input name="label" defaultValue={selectedNode?.data.label} required />
                  </label>
                  <button type="submit">保存</button>
                  <button type="button" onClick={closeModal}>取消</button>
                </form>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Wrap with ReactFlowProvider
export default function SkillTreeWrapper() {
  return (
    <ReactFlowProvider>
      <SkillTree />
    </ReactFlowProvider>
  );
}

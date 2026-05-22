/**
 * Parse Mermaid mindmap text into a tree structure.
 * The format is:
 * mindmap
 *   root((Root Node))
 *     Child 1
 *       Grandchild 1
 *       Grandchild 2
 *     Child 2
 *
 * Indentation: 2 spaces per level.
 */
export function parseMermaidToTree(text) {
  const lines = text.split('\n').filter(line => line.trim() !== '');
  if (lines.length === 0) return null;

  // First line should be 'mindmap'
  if (!lines[0].trim().startsWith('mindmap')) {
    throw new Error('Mermaid mindmap should start with "mindmap"');
  }

  // Remove the first line (mindmap) and any leading/trailing empty lines
  const dataLines = lines.slice(1);
  if (dataLines.length === 0) return null;

  // Helper to get indentation level (count of leading spaces)
  function getIndentation(line) {
    const match = line.match(/^(\s*)/);
    return match ? match[1].length : 0;
  }

  // Helper to extract node text, removing any Mermaid syntax like ((...))
  function getNodeText(line) {
    const trimmed = line.trim();
    // Remove leading/trailing spaces and potential Mermaid syntax
    // For root: root((text))
    if (trimmed.startsWith('root((')) {
      const match = trimmed.match(/^root\(\((.*)\)\)$/);
      if (match) return match[1];
    }
    // For other nodes, just return the trimmed text
    return trimmed;
  }

  // Build tree using a stack
  const root = {
    id: 'root',
    label: getNodeText(dataLines[0]),
    children: [],
    level: 0,
  };
  const stack = [{ node: root, indent: getIndentation(dataLines[0]) }];

  for (let i = 1; i < dataLines.length; i++) {
    const line = dataLines[i];
    const indent = getIndentation(line);
    const text = getNodeText(line);

    // Create new node
    const newNode = {
      id: `node-${i}`,
      label: text,
      children: [],
      level: indent / 2, // assuming 2 spaces per level
    };

    // Find parent: pop stack until we find a node with smaller indent
    while (stack.length > 0 && stack[stack.length - 1].indent >= indent) {
      stack.pop();
    }

    if (stack.length === 0) {
      throw new Error(`Invalid indentation at line ${i + 2}: ${line}`);
    }

    const parent = stack[stack.length - 1].node;
    parent.children.push(newNode);

    // Push current node onto stack
    stack.push({ node: newNode, indent });
  }

  return root;
}

/**
 * Convert tree structure to ReactFlow nodes and edges.
 * Options:
 * - layout: 'TB' (top-bottom) or 'LR' (left-right) - default 'TB'
 * - nodeWidth: number - default 200
 * - nodeHeight: number - default 50
 */
export function treeToReactFlow(tree, options = {}) {
  const { layout = 'TB', nodeWidth = 200, nodeHeight = 50 } = options;
  const nodes = [];
  const edges = [];
  let nodeIdCounter = 0;

  function traverse(node, parentId = null, depth = 0) {
    const nodeId = node.id || `node-${nodeIdCounter++}`;
    const x = layout === 'TB' ? depth * 300 : 0; // placeholder, will be overridden by layout algorithm
    const y = layout === 'TB' ? 0 : depth * 200;

    nodes.push({
      id: nodeId,
      type: 'skillNode',
      data: { label: node.label, children: node.children },
      position: { x, y },
      style: {
        width: nodeWidth,
        height: nodeHeight,
      },
    });

    if (parentId) {
      edges.push({
        id: `edge-${parentId}-${nodeId}`,
        source: parentId,
        target: nodeId,
        type: layout === 'TB' ? 'smoothstep' : 'smoothstep',
        animated: true,
        style: { stroke: '#555' },
      });
    }

    node.children.forEach(child => {
      traverse(child, nodeId, depth + 1);
    });
  }

  traverse(tree);
  return { nodes, edges };
}

/**
 * Calculate tree layout using simple algorithm.
 * This is a placeholder; we can use dagre or reactflow's layout later.
 */
export function calculateTreeLayout(nodes, edges, options = {}) {
  const { direction = 'TB', nodeWidth = 200, nodeHeight = 50, levelSpacing = 150, siblingSpacing = 50 } = options;

  // For now, just assign positions based on depth.
  // We'll need a proper layout algorithm for a polished look.
  // This is a simple placeholder.
  const nodesByDepth = {};
  nodes.forEach(node => {
    const depth = node.data?.level || 0;
    if (!nodesByDepth[depth]) nodesByDepth[depth] = [];
    nodesByDepth[depth].push(node);
  });

  Object.keys(nodesByDepth).forEach(depth => {
    const nodesInDepth = nodesByDepth[depth];
    const totalWidth = nodesInDepth.length * (nodeWidth + siblingSpacing);
    nodesInDepth.forEach((node, index) => {
      if (direction === 'TB') {
        node.position = {
          x: index * (nodeWidth + siblingSpacing) - totalWidth / 2,
          y: depth * (nodeHeight + levelSpacing),
        };
      } else {
        node.position = {
          x: depth * (nodeWidth + levelSpacing),
          y: index * (nodeHeight + siblingSpacing) - totalWidth / 2,
        };
      }
    });
  });

  return { nodes, edges };
}

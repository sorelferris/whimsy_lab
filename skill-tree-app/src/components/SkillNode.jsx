import { memo } from 'react';
import { Handle, Position } from 'reactflow';

const SkillNode = ({ data, isConnectable, selected }) => {
  const { label, children = [], onToggle, isExpanded, onEdit, onDelete, onViewDetails } = data;
  const hasChildren = children.length > 0;

  return (
    <div className={`skill-node ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      <div className="skill-node-content">
        <div className="skill-node-header">
          {hasChildren && (
            <button
              className="toggle-btn"
              onClick={(e) => {
                e.stopPropagation();
                onToggle?.();
              }}
              title={isExpanded ? '折叠' : '展开'}
            >
              {isExpanded ? '▼' : '▶'}
            </button>
          )}
          <span className="skill-node-label">{label}</span>
          <div className="skill-node-actions">
            <button
              className="action-btn"
              onClick={(e) => {
                e.stopPropagation();
                onViewDetails?.();
              }}
              title="查看详情"
            >
              👁
            </button>
            <button
              className="action-btn"
              onClick={(e) => {
                e.stopPropagation();
                onEdit?.();
              }}
              title="编辑"
            >
              ✏
            </button>
            <button
              className="action-btn delete"
              onClick={(e) => {
                e.stopPropagation();
                onDelete?.();
              }}
              title="删除"
            >
              ❌
            </button>
          </div>
        </div>
        {hasChildren && !isExpanded && (
          <div className="skill-node-children-count">
            {children.length} 个子节点
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
    </div>
  );
};

export default memo(SkillNode);

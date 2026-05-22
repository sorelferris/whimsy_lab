import SkillTree from './components/SkillTree';
import './App.css';

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>机器人算法开发技能树</h1>
        <p>交互式技能树 - 点击节点展开/折叠，拖拽节点重新排列，查看详情</p>
      </header>
      <main className="app-main">
        <SkillTree />
      </main>
    </div>
  );
}

export default App;

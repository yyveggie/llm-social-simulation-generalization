const MAP = { running: "run", succeeded: "ok", failed: "bad", stopped: "bad", paused: "paused" };
const ZH = { running: "运行中", succeeded: "成功", failed: "失败", stopped: "已停止", paused: "已暂停" };

export default function StatusPill({ status }) {
  return <span className={"pill " + (MAP[status] || "")}>{ZH[status] || status}</span>;
}

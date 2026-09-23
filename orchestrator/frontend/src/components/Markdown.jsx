function renderInline(text, kp) {
  const out = [];
  let rest = String(text);
  const re = /\*\*([^*]+)\*\*|`([^`]+)`/;
  let i = 0;
  let m = re.exec(rest);
  while (m) {
    if (m.index > 0) out.push(rest.slice(0, m.index));
    if (m[1] !== undefined) out.push(<strong key={kp + "b" + i}>{m[1]}</strong>);
    else out.push(<code key={kp + "c" + i}>{m[2]}</code>);
    rest = rest.slice(m.index + m[0].length);
    i += 1;
    m = re.exec(rest);
  }
  if (rest) out.push(rest);
  return out;
}

const isSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(l);

function cells(l) {
  let s = l.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((c) => c.trim());
}

export default function Markdown({ text, className }) {
  const lines = String(text || "").split("\n");
  const out = [];
  let i = 0;
  let k = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trim() === "") {
      i += 1;
      continue;
    }

    if (line.includes("|") && i + 1 < lines.length && isSep(lines[i + 1])) {
      const head = cells(line);
      const body = [];
      i += 2;
      while (i < lines.length && lines[i].trim() !== "" && lines[i].includes("|")) {
        body.push(cells(lines[i]));
        i += 1;
      }
      out.push(
        <table className="md-table" key={"t" + k}>
          <thead>
            <tr>{head.map((h, hi) => <th key={hi}>{renderInline(h, "t" + k + "h" + hi)}</th>)}</tr>
          </thead>
          <tbody>
            {body.map((r, ri) => (
              <tr key={ri}>{r.map((c, ci) => <td key={ci}>{renderInline(c, "t" + k + "r" + ri + "c" + ci)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      );
      k += 1;
      continue;
    }

    const hm = /^(#{1,6})\s+(.*)$/.exec(line);
    if (hm) {
      out.push(
        <div className={"md-h md-h" + hm[1].length} key={"h" + k}>
          {renderInline(hm[2], "h" + k)}
        </div>
      );
      k += 1;
      i += 1;
      continue;
    }

    if (/^\s*[-•]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-•]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-•]\s+/, ""));
        i += 1;
      }
      out.push(
        <ul className="md-ul" key={"u" + k}>
          {items.map((it, ii) => <li key={ii}>{renderInline(it, "u" + k + "i" + ii)}</li>)}
        </ul>
      );
      k += 1;
      continue;
    }

    const para = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^\s*[-•]\s+/.test(lines[i]) &&
      !/^#{1,6}\s+/.test(lines[i]) &&
      !(lines[i].includes("|") && i + 1 < lines.length && isSep(lines[i + 1]))
    ) {
      para.push(lines[i].trim());
      i += 1;
    }
    out.push(
      <p className="md-p" key={"p" + k}>
        {para.map((ln, li) => (
          <span key={li}>
            {li > 0 ? <br /> : null}
            {renderInline(ln, "p" + k + "l" + li)}
          </span>
        ))}
      </p>
    );
    k += 1;
  }

  return <div className={"md" + (className ? " " + className : "")}>{out}</div>;
}

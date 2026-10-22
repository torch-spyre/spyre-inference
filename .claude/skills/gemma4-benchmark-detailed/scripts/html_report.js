/* Copyright 2026 The Spyre-Inference Authors. Licensed under the Apache License, Version 2.0. */
(function () {
  var tip = document.getElementById("tip");
  var MINUS = "−";
  function num(v, d) { return v.toFixed(d); }
  function pill(r) {
    if (r.change == null) return "";
    var text, flat = r.overlap;
    if (r.vs_hf) text = r.ratio.toFixed(2) + "× " + r.vs;
    else text = (r.change >= 0 ? "+" : MINUS) + Math.abs(r.change).toFixed(1) + "% vs " + r.vs;
    var span = document.createElement("span");
    span.className = "delta" + (flat ? " flat" : "");
    span.textContent = text;
    return span.outerHTML;
  }
  function layout(bars, rows) {
    var max = 0;
    rows.forEach(function (r) { if (r.value > max) max = r.value; });
    var avail = bars.clientWidth - 96 - 230;
    if (avail < 60) avail = Math.max(bars.clientWidth - 96 - 150, 30);
    bars.querySelectorAll(".bar").forEach(function (el, i) {
      var v = rows[i].value;
      el.style.width = v ? Math.round(avail * v / max) + "px" : "2px";
    });
  }
  document.querySelectorAll(".charts[data-chart]").forEach(function (host) {
    var data = CHARTS[host.dataset.chart] || [];
    data.forEach(function (d) {
      var panel = document.createElement("div");
      panel.className = "panel";
      var pl = document.createElement("div");
      pl.className = "pl";
      pl.textContent = d.label;
      var small = document.createElement("small");
      small.textContent = d.unit + " · " + d.sub;
      pl.appendChild(small);
      panel.appendChild(pl);
      var bars = document.createElement("div");
      bars.className = "bars";
      d.rows.forEach(function (r) {
        var row = document.createElement("div");
        row.className = "row";
        row.tabIndex = 0;
        var val = r.value == null ? "FAILED" : num(r.value, d.digits);
        var reps = r.reps.map(function (x) { return num(x, d.digits); }).join(", ");
        var change = "";
        if (r.change != null) {
          change = "\n" + (r.vs_hf ? r.ratio.toFixed(2) + "× " : (r.change >= 0 ? "+" : MINUS) + Math.abs(r.change).toFixed(1) + "% vs ") + r.vs +
            (r.overlap ? " (replicate ranges overlap)" : " (replicate ranges do not overlap)");
        }
        row.dataset.tip = r.name + ": " + val + " " + d.unit + "\nreplicates: " + (reps || "-") + change;
        row.setAttribute("aria-label", r.name + " " + val + " " + d.unit);
        var nameSpan = document.createElement("span");
        nameSpan.className = "name";
        nameSpan.textContent = r.short;
        row.appendChild(nameSpan);
        var barDiv = document.createElement("div");
        barDiv.className = "bar";
        barDiv.style.background = r.color;
        row.appendChild(barDiv);
        var valSpan = document.createElement("span");
        valSpan.className = "val";
        var b = document.createElement("b");
        b.textContent = val;
        valSpan.appendChild(b);
        valSpan.appendChild(document.createTextNode(" " + d.unit.split(" ")[0]));
        valSpan.insertAdjacentHTML('beforeend', pill(r));
        row.appendChild(valSpan);
        bars.appendChild(row);
      });
      panel.appendChild(bars);
      host.appendChild(panel);
      requestAnimationFrame(function () { layout(bars, d.rows); });
      window.addEventListener("resize", function () { layout(bars, d.rows); });
    });
    function show(e) {
      var t = e.target.closest && e.target.closest(".row");
      if (!t) return;
      tip.textContent = t.dataset.tip;
      tip.style.whiteSpace = "pre-line";
      tip.hidden = false;
      var rect = t.getBoundingClientRect();
      var x = (e.clientX || rect.left + 20) + 14;
      var y = (e.clientY || rect.top) + 14;
      tip.style.left = Math.max(8, Math.min(x, window.innerWidth - tip.offsetWidth - 12)) + "px";
      tip.style.top = y + "px";
    }
    host.addEventListener("mousemove", show);
    host.addEventListener("focusin", show);
    host.addEventListener("mouseleave", function () { tip.hidden = true; });
    host.addEventListener("focusout", function () { tip.hidden = true; });
  });
})();
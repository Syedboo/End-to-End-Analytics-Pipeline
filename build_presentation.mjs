import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUTPUT_ROOT =
  process.env.PRINTING_ANALYTICS_OUTPUT_ROOT ||
  "C:\\Users\\e16013172\\.codex\\visualizations\\2026\\07\\28\\019fa8e2-6029-79f0-a635-3ce15aa46151\\printing_analytics_outputs";
const TMP_DIR =
  process.env.PRINTING_ANALYTICS_PPT_TMP ||
  "C:\\Users\\e16013172\\.codex\\visualizations\\2026\\07\\28\\019fa8e2-6029-79f0-a635-3ce15aa46151\\ppt_build_tmp";
const FINAL_PPTX =
  process.env.PRINTING_ANALYTICS_FINAL_PPTX ||
  "C:\\Users\\e16013172\\.codex\\visualizations\\2026\\07\\28\\019fa8e2-6029-79f0-a635-3ce15aa46151\\wg_baird_board_presentation.pptx";

const TABLES_DIR = path.join(OUTPUT_ROOT, "tables");
const REPORTS_DIR = path.join(OUTPUT_ROOT, "reports");
const SLIDE_SIZE = { width: 1280, height: 720 };
const COLORS = {
  canvas: "#FFFFFF",
  ink: "#000000",
  muted: "#59606A",
  panel: "#EDEDED",
  rule: "#B8BCC4",
  accent: "#6DCBF4",
  accentStrong: "#3D8DFF",
  paleAccent: "#D0EDFA",
};

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
}

function gbp(value) {
  return `GBP ${(Number(value) / 1_000_000).toFixed(2)}m`;
}

function pct(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function parseCsv(text) {
  const rows = [];
  let current = "";
  let row = [];
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') {
      current += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(current);
      current = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(current);
      if (row.some((cell) => cell.length > 0)) rows.push(row);
      row = [];
      current = "";
    } else {
      current += char;
    }
  }
  if (current.length || row.length) {
    row.push(current);
    rows.push(row);
  }
  const [header, ...body] = rows;
  return body.map((values) =>
    Object.fromEntries(header.map((key, index) => [key, values[index] ?? ""])),
  );
}

async function readCsv(name) {
  const text = await fs.readFile(path.join(TABLES_DIR, name), "utf8");
  return parseCsv(text);
}

function addText(slide, text, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontSize: style.fontSize ?? 24,
    bold: style.bold ?? false,
    color: style.color ?? COLORS.ink,
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "top",
  };
  return box;
}

function addRule(slide, left, top, width) {
  slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height: 2 },
    fill: COLORS.rule,
    line: { style: "solid", fill: COLORS.rule, width: 0 },
  });
}

function addFooter(slide, number) {
  addText(
    slide,
    String(number).padStart(2, "0"),
    { left: 1184, top: 659, width: 54, height: 26 },
    { fontSize: 13, alignment: "right", color: COLORS.muted },
  );
}

function addSlideTitle(slide, title, number) {
  addText(slide, title, { left: 41, top: 36, width: 1120, height: 110 }, {
    fontSize: 39,
    bold: false,
    color: COLORS.ink,
  });
  addFooter(slide, number);
}

function addPanel(slide, position, fill = COLORS.panel) {
  return slide.shapes.add({
    geometry: "rect",
    position,
    fill,
    line: { style: "solid", fill: fill, width: 0 },
  });
}

function addMetric(slide, value, label, left, top, width = 250) {
  addText(slide, value, { left, top, width, height: 54 }, {
    fontSize: 34,
    bold: true,
    color: COLORS.ink,
  });
  addText(slide, label, { left, top: top + 55, width, height: 54 }, {
    fontSize: 18,
    color: COLORS.muted,
  });
}

function setSources(slide, lines) {
  slide.speakerNotes.textFrame.setText([
    "[Sources]",
    ...lines.map((line) => `- ${line}`),
  ]);
  slide.speakerNotes.setVisible(true);
}

function addTable(slide, values, left, top, width, height, columnWidths) {
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left,
    top,
    width,
    height,
    values,
    ...(columnWidths ? { columnWidths } : {}),
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: COLORS.rule, width: 1 });
  values[0].forEach((_, index) => {
    const cell = table.getCell(0, index);
    cell.fill = COLORS.ink;
    cell.text.style = { fontSize: 15, bold: true, color: COLORS.canvas };
  });
  for (let row = 1; row < values.length; row += 1) {
    for (let col = 0; col < values[0].length; col += 1) {
      table.getCell(row, col).text.style = { fontSize: 14, color: COLORS.ink };
    }
  }
  return table;
}

async function main() {
  await fs.mkdir(TMP_DIR, { recursive: true });
  await fs.writeFile(
    path.join(TMP_DIR, "source-notes.txt"),
    [
      "W&G Baird sample workbook, Master Plain (Anon), 6,355 jobs.",
      "Generated project outputs in printing_analytics_outputs/tables and reports.",
      "No external research sources used in the deck.",
    ].join("\n"),
    "utf8",
  );

  const topCustomers = await readCsv("top_customers_by_va.csv");
  const industries = await readCsv("top_industries_by_va.csv");
  const workTypes = await readCsv("top_work_types_by_va.csv");
  const products = await readCsv("top_product_types_by_va.csv");
  const regions = await readCsv("top_regions_by_va.csv");
  const reps = await readCsv("top_sales_representatives_by_va.csv");
  const relationships = await readCsv("relationship_analysis.csv");
  const modelPerformance = await readCsv("model_performance.csv");
  const featureImportance = await readCsv("feature_importance.csv");
  const churn = await readCsv("customer_reorder_churn_opportunities.csv");

  const followUps = churn.filter((row) =>
    ["High follow-up priority", "Due for reorder"].includes(row["Churn Risk"]),
  );
  const followUpVa = followUps.reduce(
    (total, row) => total + Number(row["Customer Lifetime VA"] || 0),
    0,
  );

  const presentation = Presentation.create({ slideSize: SLIDE_SIZE });

  const slide1 = presentation.slides.add();
  slide1.background.fill = COLORS.canvas;
  addText(slide1, "W&G Baird", { left: 41, top: 41, width: 420, height: 60 }, {
    fontSize: 32,
    color: COLORS.muted,
  });
  addText(
    slide1,
    "Printing analytics identifies where value is created",
    { left: 41, top: 183, width: 992, height: 250 },
    { fontSize: 67, color: COLORS.ink },
  );
  addText(
    slide1,
    "6,355 historical jobs analysed through a refreshable Python decision-support pipeline.",
    { left: 41, top: 505, width: 720, height: 95 },
    { fontSize: 28, color: COLORS.ink },
  );
  addRule(slide1, 41, 638, 520);
  setSources(slide1, [
    "data/raw/sample_dataset.xlsx and outputs/reports/business_report.html.",
  ]);

  const slide2 = presentation.slides.add();
  slide2.background.fill = COLORS.canvas;
  addSlideTitle(slide2, "The artefact turns raw jobs into board-ready decisions", 2);
  const steps = [
    ["Clean", "Types, dates, missingness, outliers"],
    ["Engineer", "Margin, productivity, lead-time and lifecycle features"],
    ["Analyse", "Rankings, trends, relationships and statistical tests"],
    ["Predict", "Regression models and feature importance for VA Amount"],
    ["Communicate", "HTML report, tables, charts and presentation assets"],
  ];
  steps.forEach((step, index) => {
    const top = 178 + index * 82;
    addText(slide2, `0${index + 1}`, { left: 72, top, width: 70, height: 38 }, {
      fontSize: 22,
      bold: true,
      color: COLORS.accentStrong,
    });
    addText(slide2, step[0], { left: 156, top: top - 2, width: 220, height: 40 }, {
      fontSize: 27,
      bold: true,
    });
    addText(slide2, step[1], { left: 390, top: top + 2, width: 680, height: 38 }, {
      fontSize: 22,
      color: COLORS.muted,
    });
    addRule(slide2, 72, top + 58, 980);
  });
  addPanel(slide2, { left: 1050, top: 176, width: 138, height: 370 }, COLORS.paleAccent);
  addText(slide2, "Live\nsystem", { left: 1066, top: 300, width: 106, height: 82 }, {
    fontSize: 24,
    bold: true,
    alignment: "center",
    color: COLORS.ink,
  });
  setSources(slide2, [
    "printing_analytics/main.py, src/data_cleaning.py, src/feature_engineering.py, src/eda.py, src/modelling.py, src/reporting.py.",
  ]);

  const slide3 = presentation.slides.add();
  slide3.background.fill = COLORS.canvas;
  addSlideTitle(slide3, "Value is concentrated in identifiable accounts and sectors", 3);
  addMetric(slide3, gbp(topCustomers[0].VA_Amount), `${topCustomers[0]["Customer Name"]} total VA`, 63, 164, 280);
  addMetric(slide3, gbp(industries[0].VA_Amount), `${industries[0].Industry} sector VA`, 374, 164, 330);
  addMetric(slide3, gbp(regions[0].VA_Amount), `${regions[0].Region} region VA`, 733, 164, 260);
  addMetric(slide3, gbp(reps[0].VA_Amount), `${reps[0].Rep} managed VA`, 1000, 164, 220);
  addTable(
    slide3,
    [
      ["Customer", "Jobs", "Revenue", "VA", "VA margin"],
      ...topCustomers.slice(0, 5).map((row) => [
        row["Customer Name"],
        row.Jobs,
        gbp(row.Revenue),
        gbp(row.VA_Amount),
        pct(row.VA_Margin),
      ]),
    ],
    63,
    330,
    1110,
    260,
    [260, 115, 230, 230, 180],
  );
  setSources(slide3, [
    "outputs/tables/top_customers_by_va.csv.",
    "outputs/tables/top_industries_by_va.csv.",
    "outputs/tables/top_regions_by_va.csv.",
    "outputs/tables/top_sales_representatives_by_va.csv.",
  ]);

  const slide4 = presentation.slides.add();
  slide4.background.fill = COLORS.canvas;
  addSlideTitle(slide4, "Litho drives the value pool; mix still matters by margin", 4);
  slide4.charts.add("bar", {
    position: { left: 62, top: 155, width: 690, height: 430 },
    categories: workTypes.map((row) => row["Work Type"]),
    series: [
      {
        name: "VA Amount GBPm",
        values: workTypes.map((row) => Number(row.VA_Amount) / 1_000_000),
        fill: COLORS.accentStrong,
      },
    ],
    hasLegend: false,
    chartFill: COLORS.canvas,
    chartLine: { style: "solid", fill: COLORS.canvas, width: 0 },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", fill: COLORS.canvas, width: 0 },
    yAxis: {
      title: "GBP millions",
      majorGridlines: { style: "solid", fill: COLORS.panel, width: 1 },
      line: { style: "solid", fill: COLORS.rule, width: 1 },
    },
    xAxis: { line: { style: "solid", fill: COLORS.rule, width: 1 } },
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 90 },
  });
  addTable(
    slide4,
    [
      ["Product type", "Jobs", "VA", "Margin"],
      ...products.slice(0, 5).map((row) => [
        row["Product Type"],
        row.Jobs,
        gbp(row.VA_Amount),
        pct(row.VA_Margin),
      ]),
    ],
    805,
    170,
    395,
    285,
    [180, 70, 85, 80],
  );
  addText(
    slide4,
    "Wide Format shows the highest average VA margin, but Litho creates by far the largest total VA because of scale.",
    { left: 805, top: 488, width: 390, height: 105 },
    { fontSize: 23, color: COLORS.ink },
  );
  setSources(slide4, [
    "outputs/tables/top_work_types_by_va.csv.",
    "outputs/tables/top_product_types_by_va.csv.",
  ]);

  const slide5 = presentation.slides.add();
  slide5.background.fill = COLORS.canvas;
  addSlideTitle(slide5, "VA is explained by price, production effort and material intensity", 5);
  addTable(
    slide5,
    [
      ["Relationship", "Pearson r"],
      ...relationships.slice(0, 5).map((row) => [
        `${row.x} vs ${row.y}`,
        Number(row.pearson_corr).toFixed(3),
      ]),
    ],
    63,
    170,
    520,
    265,
    [350, 140],
  );
  addTable(
    slide5,
    [
      ["Model", "MAE", "RMSE", "R2"],
      ...modelPerformance.slice(0, 3).map((row) => [
        row.model,
        Number(row.MAE).toFixed(0),
        Number(row.RMSE).toFixed(0),
        Number(row.R2).toFixed(3),
      ]),
    ],
    650,
    170,
    520,
    195,
    [190, 105, 105, 95],
  );
  addText(
    slide5,
    `Top model driver: ${featureImportance[0].feature}. The high R2 should be read as evidence of strong accounting and pricing structure, not as a substitute for commercial judgement.`,
    { left: 650, top: 405, width: 520, height: 126 },
    { fontSize: 23, color: COLORS.ink },
  );
  addPanel(slide5, { left: 63, top: 486, width: 520, height: 80 }, COLORS.paleAccent);
  addText(
    slide5,
    "Pricing and estimating decisions should be reviewed where markup is weak and cost-to-sales intensity is high.",
    { left: 86, top: 505, width: 474, height: 48 },
    { fontSize: 21, color: COLORS.ink },
  );
  setSources(slide5, [
    "outputs/tables/relationship_analysis.csv.",
    "outputs/tables/model_performance.csv.",
    "outputs/tables/feature_importance.csv.",
  ]);

  const slide6 = presentation.slides.add();
  slide6.background.fill = COLORS.canvas;
  addText(slide6, "Board actions", { left: 41, top: 41, width: 320, height: 60 }, {
    fontSize: 32,
    color: COLORS.muted,
  });
  addText(
    slide6,
    "Use the system weekly to protect value, improve pricing and trigger follow-up",
    { left: 41, top: 170, width: 1000, height: 220 },
    { fontSize: 57, color: COLORS.ink },
  );
  const actions = [
    ["Protect", "Give top-VA customers and sectors named account plans."],
    ["Price", "Review low-margin or negative-markup repeat work before quote renewal."],
    ["Improve", "Target jobs where labour, paper, purchases or press hours dilute VA."],
    ["Follow up", `${followUps.length} overdue/due customers represent ${gbp(followUpVa)} historical VA.`],
  ];
  actions.forEach((row, index) => {
    const left = 63 + index * 292;
    addText(slide6, row[0], { left, top: 472, width: 230, height: 40 }, {
      fontSize: 25,
      bold: true,
      color: COLORS.ink,
    });
    addText(slide6, row[1], { left, top: 518, width: 235, height: 90 }, {
      fontSize: 18,
      color: COLORS.muted,
    });
  });
  addRule(slide6, 41, 635, 720);
  setSources(slide6, [
    "outputs/tables/customer_reorder_churn_opportunities.csv.",
    "outputs/reports/business_report.html.",
  ]);

  const renderDir = path.join(TMP_DIR, "rendered_slides");
  await fs.rm(renderDir, { recursive: true, force: true });
  await fs.mkdir(renderDir, { recursive: true });
  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(
      path.join(renderDir, `${stem}.png`),
      await presentation.export({ slide, format: "png", scale: 1 }),
    );
    await fs.writeFile(
      path.join(renderDir, `${stem}.layout.json`),
      await (await slide.export({ format: "layout" })).text(),
      "utf8",
    );
  }
  await writeBlob(
    path.join(TMP_DIR, "deck-montage.webp"),
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );
  const inspection = await presentation.inspect({
    kind: "slide,textbox,shape,table,chart,notes",
    maxChars: 12000,
  });
  await fs.writeFile(
    path.join(TMP_DIR, "deck-inspect.ndjson"),
    typeof inspection === "string" ? inspection : JSON.stringify(inspection, null, 2),
    "utf8",
  );

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(`Saved ${FINAL_PPTX}`);
  console.log(`Rendered slides ${renderDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

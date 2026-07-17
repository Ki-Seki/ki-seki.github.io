(() => {
  "use strict";

  const bi = (zh, en) => ({ zh, en });
  const BASE_PATH = "/features/hypothesis-testing-guide/";
  const UI_TEXT = {
    zh: {
      skipToContent: "跳到主要内容", brandTitle: "假设检验指南", brandSubtitle: "从研究问题到可靠结论",
      navFinder: "方法选择", navHelp: "帮助", primaryNavigation: "主要导航",
      footerNote: "统计方法帮助你量化证据，但不能替代清晰的研究设计。", footerChecklist: "查看报告检查表",
      switchLanguage: "Switch to English", languageCode: "EN", guideTitle: "从问题出发，选对统计方法",
      guideLead: "回答几个关于结果变量、研究设计和推断目标的问题，或直接搜索完整方法库。",
      findMethodTitle: "找到合适的统计方法", findMethodLead: "先选择查找方式：沿研究问题树判断，或按名称和条件搜索完整方法库。",
      modeLabel: "选择查找方式", guideModeTitle: "沿选择树判断", guideModeDesc: "不知道方法名称？从研究目标和数据结构沿树找到候选。",
      libraryModeTitle: "搜索方法库", libraryModeDesc: "已有名称或条件？搜索别名，或按研究设计筛选全部方法。", currentMode: "当前模式", openMethodShort: "查看公式与计算", alternativeMethod: "替代方法",
      finder: "帮我选择", library: "浏览方法", searchLabel: "搜索方法与概念", searchPlaceholder: "输入名称、场景、公式或代码别名，例如 paired_wilcoxon",
      searchHint: "按 / 或 Ctrl+K 聚焦；Tab 浏览结果，Esc 清空。", noResults: "没有匹配结果。试试更通用的关键词。",
      methods: "统计方法", concepts: "概念与效应量", yourConditions: "你的条件", startOver: "重新开始", previous: "上一步",
      recommendation: "推荐方法", whyThis: "为什么推荐", openMethod: "查看公式与完整计算", allMethods: "全部方法",
      filterOutcome: "结果类型", filterDesign: "设计", filterGroups: "组数", filterKind: "方法类别", clearFilters: "清除筛选", clearSearch: "清空搜索",
      continuous: "连续", categorical: "分类", ordinal: "有序", count: "计数", any: "不限", independent: "独立", paired: "配对/重复",
      one: "一组", two: "两组", many: "三组以上", comparison: "比较", association: "关联/模型", diagnostic: "前提检查",
      methodLibrary: "方法库", background: "背景与直觉", suitability: "何时使用", hypotheses: "假设", assumptions: "数据结构与前提",
      formula: "公式与符号", example: "逐步计算", inference: "p 值与推断", effect: "置信区间与效应量", edgeCases: "特殊情况",
      reporting: "报告模板", python: "Python 复现", related: "相关方法", references: "参考资料", contents: "本页目录",
      useWhen: "适用", avoidWhen: "不适用", nullHypothesis: "原假设 H₀", alternativeHypothesis: "备择假设 H₁",
      symbols: "符号表", calculationSteps: "代入步骤", result: "结果", approximate: "近似结果", copy: "复制", copied: "已复制",
      backToLibrary: "返回方法库", helpTitle: "统计推断帮助中心", helpLead: "先理解研究设计与估计目标，再选择检验；不要让单个前提检验的 p 值替你做决定。",
      glossary: "中英文术语表", checklist: "报告检查表", updated: "内容审校：2026-07-17", notFound: "没有找到这个方法",
      notFoundText: "URL 中的方法 ID 不存在。你可以返回方法库搜索。", searchResults: "搜索结果", resultCount: "个结果"
    },
    en: {
      skipToContent: "Skip to main content", brandTitle: "Hypothesis Testing Guide", brandSubtitle: "From research questions to reliable conclusions",
      navFinder: "Find a method", navHelp: "Help", primaryNavigation: "Primary navigation",
      footerNote: "Statistical methods quantify evidence; they do not replace a clear study design.", footerChecklist: "Open the reporting checklist",
      switchLanguage: "切换到中文", languageCode: "中", guideTitle: "Start with the question, then choose the method",
      guideLead: "Answer a few questions about the outcome, design, and estimand, or search the complete method library.",
      findMethodTitle: "Find the right statistical method", findMethodLead: "Choose how to find it: follow the research-question tree or search and filter the complete method library.",
      modeLabel: "Choose how to find a method", guideModeTitle: "Follow the decision tree", guideModeDesc: "Not sure of the method? Follow the research goal and data structure to a candidate.",
      libraryModeTitle: "Search the method library", libraryModeDesc: "Know a name or condition? Search aliases or filter the full library by study design.", currentMode: "Current mode", openMethodShort: "View formulas & calculation", alternativeMethod: "Alternative method",
      finder: "Help me choose", library: "Browse methods", searchLabel: "Search methods and concepts", searchPlaceholder: "Search a name, use case, formula, or code alias, e.g. paired_wilcoxon",
      searchHint: "Press / or Ctrl+K to focus; use Tab to browse results; Esc clears.", noResults: "No matches. Try a broader term.",
      methods: "Statistical methods", concepts: "Concepts and effect sizes", yourConditions: "Your conditions", startOver: "Start over", previous: "Back",
      recommendation: "Recommended method", whyThis: "Why it fits", openMethod: "See formulas and full calculation", allMethods: "All methods",
      filterOutcome: "Outcome", filterDesign: "Design", filterGroups: "Groups", filterKind: "Method type", clearFilters: "Clear filters", clearSearch: "Clear search",
      continuous: "Continuous", categorical: "Categorical", ordinal: "Ordinal", count: "Count", any: "Any", independent: "Independent", paired: "Paired/repeated",
      one: "One", two: "Two", many: "Three or more", comparison: "Comparison", association: "Association/model", diagnostic: "Assumption check",
      methodLibrary: "Method library", background: "Background and intuition", suitability: "When to use it", hypotheses: "Hypotheses", assumptions: "Data structure and assumptions",
      formula: "Formula and notation", example: "Worked calculation", inference: "p-value and inference", effect: "Confidence intervals and effect size", edgeCases: "Edge cases",
      reporting: "Reporting template", python: "Python reproduction", related: "Related methods", references: "References", contents: "On this page",
      useWhen: "Use when", avoidWhen: "Avoid when", nullHypothesis: "Null hypothesis H₀", alternativeHypothesis: "Alternative hypothesis H₁",
      symbols: "Notation", calculationSteps: "Substitution steps", result: "Result", approximate: "Approximate result", copy: "Copy", copied: "Copied",
      backToLibrary: "Back to library", helpTitle: "Statistical inference help center", helpLead: "Understand the design and estimand before choosing a test; do not let one assumption-test p-value decide for you.",
      glossary: "Chinese–English glossary", checklist: "Reporting checklist", updated: "Content reviewed: 2026-07-17", notFound: "Method not found",
      notFoundText: "The method ID in the URL does not exist. Return to the library and search.", searchResults: "Search results", resultCount: "results"
    }
  };

  const OFFICIAL_REFS = {
    scipy: { label: bi("SciPy stats 官方参考", "SciPy stats reference"), url: "https://docs.scipy.org/doc/scipy/reference/stats.html" },
    scipyNorm: { label: bi("SciPy 标准正态分布接口", "SciPy normal-distribution API"), url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html" },
    scipyBinom: { label: bi("SciPy 精确二项检验接口", "SciPy exact binomial-test API"), url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html" },
    statsmodels: { label: bi("statsmodels 官方统计参考", "statsmodels statistical reference"), url: "https://www.statsmodels.org/stable/stats.html" },
    proportionsZ: { label: bi("statsmodels 比例 Z 检验接口", "statsmodels proportions Z-test API"), url: "https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html" },
    proportionsCI: { label: bi("statsmodels 两独立比例区间接口", "statsmodels two-independent-proportions CI API"), url: "https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.confint_proportions_2indep.html" },
    nist: { label: bi("NIST/SEMATECH 统计方法手册", "NIST/SEMATECH e-Handbook of Statistical Methods"), url: "https://www.itl.nist.gov/div898/handbook/" }
  };

  function makeMethod(config) {
    const defaults = {
      avoidWhen: bi("研究设计与该方法不匹配、观测并非所需的独立或配对结构，或关键模型前提明显失效时。", "Avoid it when the design does not match, observations lack the required independence or pairing, or core model assumptions clearly fail."),
      assumptions: bi(["研究问题、分析单位和比较方向在看数据前定义。", "观测值按研究设计独立；配对方法只要求不同配对之间独立。", "报告缺失值、异常值以及任何排除规则。"], ["Define the question, unit of analysis, and contrast before inspecting outcomes.", "Observations follow the design's independence structure; paired methods require independence between pairs.", "Report missing values, outliers, and all exclusion rules."]),
      formulaNotes: bi("公式给出常用双侧检验；单侧检验必须在看数据前由方向性假设决定。", "Formulas show the usual two-sided test. A one-sided alternative must be prespecified."),
      inference: bi("在 H₀ 下，用统计量的零分布计算观测值同样或更极端的概率。p 值不是 H₀ 为真的概率。", "Under H₀, compute the probability of a statistic at least as extreme as observed. The p-value is not the probability that H₀ is true."),
      ci: bi("同时报告与检验相对应的置信区间；区间展示方向、精度以及与零效应的相容程度。", "Report the confidence interval corresponding to the test; it conveys direction, precision, and compatibility with the null value."),
      effect: bi("同时报告与问题匹配的效应量，不要只报告 p 值。", "Report an effect size aligned with the estimand rather than a p-value alone."),
      edgeCases: bi(["缺失值按预先说明的规则处理，不把缺失自动当作零。", "小样本、并列值或边界数据可能需要精确算法、置换法或稳健区间。", "多次检验时控制家族错误率或假发现率。"], ["Handle missing values using a prespecified rule; never silently replace them with zero.", "Small samples, ties, or boundary data may require exact algorithms, permutation methods, or robust intervals.", "Control family-wise error or false discovery rate when testing many hypotheses."]),
      report: bi("报告估计值、95% CI、检验统计量、自由度（若适用）、精确 p 值、样本量与效应量。", "Report the estimate, 95% CI, test statistic, degrees of freedom when applicable, exact p-value, sample size, and effect size."),
      references: [OFFICIAL_REFS.scipy, OFFICIAL_REFS.nist],
      related: []
    };
    const c = { ...defaults, ...config };
    return {
      id: c.id, family: c.family, category: c.category, outcome: c.outcome, design: c.design, groups: c.groups, kind: c.kind,
      aliases: [...new Set([c.id, ...(c.aliases || [])])], formulas: c.formulas, symbols: c.symbols, example: c.example,
      python: c.python, related: c.related, references: c.references,
      zh: {
        name: c.name.zh, short: c.short.zh, background: (c.background || c.short).zh, useWhen: c.useWhen.zh, avoidWhen: c.avoidWhen.zh,
        hypotheses: { h0: c.h0.zh, h1: c.h1.zh }, assumptions: c.assumptions.zh, formulaNotes: c.formulaNotes.zh,
        inference: c.inference.zh, ci: c.ci.zh, effect: c.effect.zh, edgeCases: c.edgeCases.zh, report: c.report.zh
      },
      en: {
        name: c.name.en, short: c.short.en, background: (c.background || c.short).en, useWhen: c.useWhen.en, avoidWhen: c.avoidWhen.en,
        hypotheses: { h0: c.h0.en, h1: c.h1.en }, assumptions: c.assumptions.en, formulaNotes: c.formulaNotes.en,
        inference: c.inference.en, ci: c.ci.en, effect: c.effect.en, edgeCases: c.edgeCases.en, report: c.report.en
      }
    };
  }

  const tSymbols = [
    { symbol: "\\bar{x}", meaning: bi("样本均值", "sample mean") },
    { symbol: "s", meaning: bi("样本标准差", "sample standard deviation") },
    { symbol: "n", meaning: bi("有效样本量", "effective sample size") }
  ];

  const METHODS = {};
  const add = (config) => { const item = makeMethod(config); METHODS[item.id] = item; };

  add({
    id: "one_sample_z", family: "parametric", category: bi("均值比较", "Mean comparison"), outcome: "continuous", design: "one", groups: "one", kind: "comparison",
    aliases: ["z test", "ztest", "z-test", "z 检验", "z检验", "mean z test", "one sample z", "one-sample z-test", "one sample mean z", "known sigma z", "known variance z", "单样本z检验", "单样本均值z检验", "均值z检验", "总体方差已知"],
    name: bi("单样本均值 Z 检验（已知总体 σ）", "One-sample mean Z-test (known population σ)"),
    short: bi("总体标准差由样本外信息确切给定时，检验一个总体均值。", "Tests one population mean when its standard deviation is known from external information."),
    background: bi("把样本均值与假设均值之差除以已知的标准误 σ/√n；H₀ 下统计量服从标准正态。", "It divides the difference between the sample and null means by the known standard error σ/√n; under H₀ the statistic is standard normal."),
    useWhen: bi("一组独立连续观测；总体 σ 在分析前由可靠外部资料确切给定；总体正态，或 n 足够大使均值的正态近似可靠。", "Use for one independent continuous sample when population σ is genuinely known before analysis and the population is normal, or n is large enough for the mean's normal approximation."),
    avoidWhen: bi("σ 是用当前样本估计的；这时应使用单样本 t 检验。小样本且总体明显非正态、观测聚类或抽样权重复杂时也不要使用简单 Z 公式。", "Do not use when σ is estimated from the current sample—use a one-sample t-test. Also avoid the simple formula for small clearly nonnormal samples, clustering, or complex survey weights."),
    h0: bi("μ=μ₀。", "μ=μ₀."), h1: bi("μ≠μ₀（双侧），或预先指定 μ>μ₀ / μ<μ₀。", "μ≠μ₀ (two-sided), or a prespecified μ>μ₀ / μ<μ₀."),
    assumptions: bi([
      "观测独立且来自目标总体；分析单位定义正确。",
      "总体标准差 σ 真正已知，且不是由本样本估计。",
      "小样本时总体近似正态；大样本依赖均值的中心极限定理，并检查极端值。"
    ], [
      "Observations are independent draws from the target population and the unit of analysis is correct.",
      "Population SD σ is genuinely known and not estimated from these data.",
      "For small n the population is approximately normal; for large n rely on the CLT for the mean and inspect extremes."
    ]),
    formulas: [
      { label: bi("标准误与 Z", "Standard error and Z"), tex: "SE(\\bar X)=\\frac{\\sigma}{\\sqrt n},\\qquad Z=\\frac{\\bar X-\\mu_0}{\\sigma/\\sqrt n}" },
      { label: bi("双侧 p 值", "Two-sided p-value"), tex: "p=2\\{1-\\Phi(|z_{obs}|)\\}" },
      { label: bi("均值与均值差区间", "Intervals for mean and mean difference"), tex: "\\bar x\\pm z_{1-\\alpha/2}\\frac{\\sigma}{\\sqrt n},\\qquad (\\bar x-\\mu_0)\\pm z_{1-\\alpha/2}\\frac{\\sigma}{\\sqrt n}" },
      { label: bi("按已知 σ 标准化", "Standardization by known σ"), tex: "d_\\sigma=\\frac{\\bar x-\\mu_0}{\\sigma}" }
    ],
    symbols: [
      { symbol: "\\bar x", meaning: bi("样本均值", "sample mean") },
      { symbol: "\\mu_0", meaning: bi("原假设均值", "null mean") },
      { symbol: "\\sigma", meaning: bi("外部已知的总体标准差", "externally known population SD") },
      { symbol: "\\Phi", meaning: bi("标准正态累计分布函数", "standard normal CDF") },
      { symbol: "n", meaning: bi("有效独立样本量", "effective independent sample size") }
    ],
    example: {
      caption: bi("25 个观测与 50 比较，外部已知 σ=5", "Twenty-five observations compared with 50, with externally known σ=5"),
      columns: bi(["观测块", "5 个原始观测"], ["Block", "Five raw observations"]),
      rows: [[1,"43, 45, 46, 47, 48"],[2,"49, 49, 50, 50, 51"],[3,"51, 52, 52, 52, 53"],[4,"53, 54, 54, 55, 55"],[5,"56, 57, 58, 59, 61"]],
      steps: bi([
        "n=25，x̄=52，μ₀=50，已知 σ=5。",
        "SE=5/√25=1。",
        "Z=(52−50)/1=2.000。",
        "双侧 p=2[1−Φ(2)]=.045500。",
        "均值 95% CI=52±1.959964×1=[50.040,53.960]；均值差 CI=[.040,3.960]；d_σ=.400。"
      ], [
        "n=25, x̄=52, μ₀=50, and known σ=5.",
        "SE=5/√25=1.",
        "Z=(52−50)/1=2.000.",
        "Two-sided p=2[1−Φ(2)]=.045500.",
        "95% CI for μ is 52±1.959964×1=[50.040,53.960]; difference CI=[.040,3.960]; d_σ=.400."
      ]),
      result: bi("在 σ=5 确实已知的模型下，均值略高于 50；区间下界非常接近 50，应避免把 p=.0455 夸大为强证据。", "Under a model with genuinely known σ=5, the mean is slightly above 50; the interval barely excludes 50, so p=.0455 is not strong evidence."),
      approximate: true
    },
    inference: bi("双侧 p 来自标准正态尾面积。已知 σ 时没有自由度估计；若 σ 来自本样本，换用 t 分布。", "The two-sided p-value is a standard-normal tail area. There is no estimated df when σ is known; if σ comes from the sample, use the t distribution."),
    ci: bi("均值 95% CI=[50.040,53.960]，与同一双侧 α=.05 Z 检验对偶；均值差 CI=[.040,3.960]。", "The 95% CI for μ is [50.040,53.960], dual to the two-sided α=.05 Z-test; the difference CI is [.040,3.960]."),
    effect: bi("原始均值差=2；按已知总体 σ 标准化 d_σ=.400。必须明确分母是外部 σ，而不是样本 SD。", "The raw mean difference is 2 and d_σ=.400 using known population σ; state that the denominator is external σ, not the sample SD."),
    edgeCases: bi([
      "现实研究中总体 σ 很少真正已知；历史估计有不确定性时，简单 Z 检验通常低估不确定性。",
      "σ 单位必须与结果相同，且应对应当前目标总体与测量流程。",
      "单侧方向、μ₀ 和 σ 必须在看结果前确定；多次查看或多重检验需校正。"
    ], [
      "Population σ is rarely truly known; treating an uncertain historical estimate as fixed usually understates uncertainty.",
      "σ must share the outcome's units and apply to the current population and measurement process.",
      "Direction, μ₀, and σ must be set before outcomes are inspected; repeated looks or multiple tests need correction."
    ]),
    report: bi("在外部已知 σ=5 的前提下，样本均值为 52.0（n=25），Z=2.000，双侧 p=.0455，均值差=2.0，95% CI [.040,3.960]，d_σ=.400。", "With externally known σ=5, the sample mean was 52.0 (n=25), Z=2.000, two-sided p=.0455, mean difference=2.0, 95% CI [.040,3.960], d_σ=.400."),
    python: "import numpy as np\nfrom scipy import stats\nx = np.array([43,45,46,47,48,49,49,50,50,51,51,52,52,52,53,53,54,54,55,55,56,57,58,59,61])\nmu0, sigma = 50, 5\nse = sigma / np.sqrt(x.size)\nz = (x.mean() - mu0) / se\np = 2 * stats.norm.sf(abs(z))\nzcrit = stats.norm.ppf(0.975)\nci_mu = (x.mean()-zcrit*se, x.mean()+zcrit*se)\nprint(z, p, ci_mu)",
    references: [OFFICIAL_REFS.scipyNorm, OFFICIAL_REFS.nist],
    related: ["one_sample_t", "shapiro_wilk"]
  });

  add({
    id: "one_sample_t", family: "parametric", category: bi("均值比较", "Mean comparison"), outcome: "continuous", design: "one", groups: "one", kind: "comparison",
    aliases: ["one sample t", "ttest_1samp", "单样本t检验"], name: bi("单样本 t 检验", "One-sample t-test"),
    short: bi("检验一个总体均值是否等于预先给定的标准值。", "Tests whether one population mean equals a prespecified value."),
    background: bi("把样本均值与标准值之差，除以该差异的标准误。差异相对抽样噪声越大，|t| 越大。", "It scales the difference between the sample mean and target by its standard error; larger signal relative to sampling noise produces larger |t|."),
    useWhen: bi("一组连续结果；目标是总体均值；观测独立，数据近似正态，或样本量足以让均值近似正态。", "Use for one independent continuous sample when the estimand is the population mean and the data, or sampling distribution of the mean, are approximately normal."),
    h0: bi("μ = μ₀。", "μ = μ₀."), h1: bi("μ ≠ μ₀（双侧），或预先指定 μ > μ₀ / μ < μ₀。", "μ ≠ μ₀ (two-sided), or a prespecified μ > μ₀ / μ < μ₀."),
    formulas: [
      { label: bi("检验统计量", "Test statistic"), tex: "t=\\frac{\\bar{x}-\\mu_0}{s/\\sqrt{n}},\\qquad df=n-1" },
      { label: bi("均值差置信区间", "CI for the mean difference"), tex: "(\\bar{x}-\\mu_0)\\pm t_{1-\\alpha/2,\\,n-1}\\frac{s}{\\sqrt{n}}" },
      { label: bi("标准化效应量", "Standardized effect"), tex: "d=\\frac{\\bar{x}-\\mu_0}{s}" }
    ], symbols: [...tSymbols, { symbol: "\\mu_0", meaning: bi("原假设中的标准值", "null reference value") }],
    example: { caption: bi("5 名学生成绩与 80 分比较", "Five scores compared with 80"), columns: bi(["学生", "成绩"], ["Student", "Score"]), rows: [[1,78],[2,82],[3,85],[4,79],[5,86]],
      steps: bi(["均值 x̄ = 82.00；样本标准差 s = 3.536。", "SE = 3.536/√5 = 1.581。", "t = (82−80)/1.581 = 1.265，df = 4。", "双侧 p ≈ 0.275；均值差 95% CI ≈ [−2.39, 6.39]。"], ["Mean x̄ = 82.00; sample SD s = 3.536.", "SE = 3.536/√5 = 1.581.", "t = (82−80)/1.581 = 1.265, df = 4.", "Two-sided p ≈ 0.275; 95% CI for the difference ≈ [−2.39, 6.39]."]), result: bi("未发现均值不同于 80 的证据；d = 0.566，但区间较宽。", "There is insufficient evidence that the mean differs from 80; d = 0.566, with a wide interval."), approximate: true },
    inference: bi("双侧 p = 2P(T₄ ≥ |1.265|) ≈ 0.275。不能把“不显著”解释成均值相等。", "The two-sided p-value is 2P(T₄ ≥ |1.265|) ≈ 0.275. A nonsignificant result does not establish equality."),
    ci: bi("示例中均值差 95% CI 约为 [−2.39, 6.39]；它同时包含零和具有实际意义的正差异。", "The example's 95% CI is approximately [−2.39, 6.39], spanning zero and practically meaningful positive effects."),
    effect: bi("Cohen's d = 0.566。小样本可报告偏差修正 Hedges' g。", "Cohen's d = 0.566; for small samples, consider bias-corrected Hedges' g."),
    report: bi("成绩均值为 82.0（SD=3.54），与 80 的差异不显著，t(4)=1.265，p=.275，均值差 95% CI [−2.39, 6.39]，d=0.57。", "Scores averaged 82.0 (SD=3.54), not significantly different from 80, t(4)=1.265, p=.275, 95% CI for the difference [−2.39, 6.39], d=0.57."),
    python: "import numpy as np\nfrom scipy import stats\nx = np.array([78, 82, 85, 79, 86])\nresult = stats.ttest_1samp(x, popmean=80)\nprint(result)\nprint(result.confidence_interval())",
    related: ["one_sample_z", "wilcoxon_signed", "shapiro_wilk"]
  });

  add({
    id: "independent_t", family: "parametric", category: bi("均值比较", "Mean comparison"), outcome: "continuous", design: "independent", groups: "two", kind: "comparison",
    aliases: ["student t", "two sample t", "ttest_ind", "pooled t"], name: bi("独立样本 t 检验", "Independent-samples t-test"),
    short: bi("在方差近似相等时比较两个独立总体的均值。", "Compares the means of two independent populations under equal variances."),
    useWhen: bi("两组观测彼此独立、结果连续、组内近似正态且总体方差可合理视为相等。", "Use for two independent continuous samples with approximately normal within-group outcomes and plausibly equal population variances."),
    h0: bi("μ₁−μ₂ = 0。", "μ₁−μ₂ = 0."), h1: bi("μ₁−μ₂ ≠ 0，或预先指定方向。", "μ₁−μ₂ ≠ 0, or a prespecified direction."),
    assumptions: bi(["两组对象相互独立，且组内观测独立。", "连续结果在各组内近似正态；严重离群值会强烈影响均值和 SD。", "经典 pooled t 假设 σ₁²=σ₂²；不确定时 Welch t 通常更安全。"], ["Groups and observations within groups are independent.", "Outcomes are approximately normal within groups; severe outliers affect means and SDs.", "The pooled test assumes σ₁²=σ₂²; Welch's test is usually safer when uncertain."]),
    formulas: [
      { label: bi("合并方差", "Pooled variance"), tex: "s_p^2=\\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}" },
      { label: bi("检验统计量", "Test statistic"), tex: "t=\\frac{\\bar{x}_1-\\bar{x}_2}{s_p\\sqrt{1/n_1+1/n_2}},\\quad df=n_1+n_2-2" },
      { label: bi("Cohen's d", "Cohen's d"), tex: "d=\\frac{\\bar{x}_1-\\bar{x}_2}{s_p}" }
    ], symbols: [...tSymbols, { symbol: "s_p", meaning: bi("合并组内标准差", "pooled within-group SD") }],
    example: { caption: bi("A、B 两组各 4 人", "Four observations in each of groups A and B"), columns: bi(["组别", "数值"], ["Group", "Values"]), rows: [["A","8, 9, 10, 11"],["B","11, 12, 13, 14"]],
      steps: bi(["x̄₁=9.5，x̄₂=12.5；两组 s=1.291。", "sₚ=1.291，SE=1.291√(1/4+1/4)=0.913。", "t=(9.5−12.5)/0.913=−3.286，df=6。", "双侧 p≈0.0167；均值差 95% CI≈[−5.23,−0.77]。"], ["x̄₁=9.5, x̄₂=12.5; both SDs are 1.291.", "sₚ=1.291 and SE=1.291√(1/4+1/4)=0.913.", "t=(9.5−12.5)/0.913=−3.286, df=6.", "Two-sided p≈0.0167; 95% CI≈[−5.23,−0.77]."]), result: bi("A 组均值较低；d=−2.32。样本极小，效应估计不精确。", "Group A has a lower mean; d=−2.32. With tiny samples, the effect estimate is imprecise."), approximate: true },
    inference: bi("p 值来自 df=6 的 t 分布。先看均值差和区间，再解释显著性。", "The p-value comes from a t distribution with df=6. Interpret the mean difference and interval before significance."),
    ci: bi("均值差的 t 区间使用与检验相同的 pooled SE。", "The t interval for the mean difference uses the same pooled standard error as the test."),
    effect: bi("示例 Cohen's d=−2.32；小样本优先补充 Hedges' g。", "Example Cohen's d=−2.32; supplement with Hedges' g for small samples."),
    report: bi("A 组（M=9.5, SD=1.29）低于 B 组（M=12.5, SD=1.29），t(6)=−3.286，p=.017，均值差 95% CI [−5.23,−0.77]，d=−2.32。", "Group A (M=9.5, SD=1.29) was lower than group B (M=12.5, SD=1.29), t(6)=−3.286, p=.017, 95% CI [−5.23,−0.77], d=−2.32."),
    python: "import numpy as np\nfrom scipy import stats\na = np.array([8, 9, 10, 11])\nb = np.array([11, 12, 13, 14])\nprint(stats.ttest_ind(a, b, equal_var=True))",
    related: ["welch_t", "mann_whitney", "levene_test"]
  });

  add({
    id: "welch_t", family: "parametric", category: bi("均值比较", "Mean comparison"), outcome: "continuous", design: "independent", groups: "two", kind: "comparison",
    aliases: ["welch's t", "unequal variance t", "ttest_ind equal_var false"], name: bi("Welch t 检验", "Welch's t-test"),
    short: bi("比较两个独立总体均值，无需假设方差相等。", "Compares two independent means without assuming equal variances."),
    useWhen: bi("两组独立连续结果；尤其是样本量或方差不同。若没有充分理由假定方差相等，可把 Welch 作为默认两样本 t 检验。", "Use for two independent continuous samples, especially with unequal sample sizes or variances; it is a sound default without strong evidence of equal variances."),
    h0: bi("μ₁−μ₂=0。", "μ₁−μ₂=0."), h1: bi("μ₁−μ₂≠0，或预先指定方向。", "μ₁−μ₂≠0, or a prespecified direction."),
    formulas: [
      { label: bi("Welch t", "Welch t"), tex: "t=\\frac{\\bar{x}_1-\\bar{x}_2}{\\sqrt{s_1^2/n_1+s_2^2/n_2}}" },
      { label: bi("Welch–Satterthwaite 自由度", "Welch–Satterthwaite degrees of freedom"), tex: "\\nu=\\frac{(s_1^2/n_1+s_2^2/n_2)^2}{(s_1^2/n_1)^2/(n_1-1)+(s_2^2/n_2)^2/(n_2-1)}" }
    ], symbols: [...tSymbols, { symbol: "\\nu", meaning: bi("修正后自由度，通常不是整数", "adjusted, often noninteger, degrees of freedom") }],
    example: { caption: bi("方差明显不同的两组", "Two groups with visibly different variances"), columns: bi(["组别", "数值"], ["Group", "Values"]), rows: [["A","8, 9, 10, 11"],["B","6, 12, 18, 24"]],
      steps: bi(["x̄₁=9.5，s₁²=1.667；x̄₂=15.0，s₂²=60。", "SE=√(1.667/4+60/4)=3.926。", "t=−5.5/3.926=−1.401。", "ν≈3.17；双侧 p≈0.25，结果不显著。"], ["x̄₁=9.5, s₁²=1.667; x̄₂=15.0, s₂²=60.", "SE=√(1.667/4+60/4)=3.926.", "t=−5.5/3.926=−1.401.", "ν≈3.17; two-sided p≈0.25, not significant."]), result: bi("尽管均值差为 −5.5，B 组波动很大，区间很宽。", "Although the mean difference is −5.5, high variation in group B makes the interval wide."), approximate: true },
    inference: bi("使用修正自由度的 t 分布；不要把 Levene 不显著当作必须改用 pooled t 的理由。", "Use the t distribution with adjusted degrees of freedom. A nonsignificant Levene test does not require switching to a pooled test."),
    ci: bi("均值差区间为 (x̄₁−x̄₂) ± t*√(s₁²/n₁+s₂²/n₂)。", "The CI is (x̄₁−x̄₂) ± t*√(s₁²/n₁+s₂²/n₂)."),
    effect: bi("可报告均值差及其 CI；标准化时明确使用哪种 SD（如平均 SD 或 Hedges' g）。", "Report the raw mean difference and CI; when standardizing, state the SD convention, such as average SD or Hedges' g."),
    report: bi("Welch t 检验未发现均值差异，t(3.17)=−1.40，p≈.25，均值差=−5.5。", "Welch's test did not detect a mean difference, t(3.17)=−1.40, p≈.25, mean difference=−5.5."),
    python: "from scipy import stats\na = [8, 9, 10, 11]\nb = [6, 12, 18, 24]\nprint(stats.ttest_ind(a, b, equal_var=False))",
    related: ["independent_t", "mann_whitney", "levene_test"]
  });

  add({
    id: "paired_t", family: "parametric", category: bi("配对比较", "Paired comparison"), outcome: "continuous", design: "paired", groups: "two", kind: "comparison",
    aliases: ["paired t", "dependent t", "ttest_rel", "配对t检验", "cohens_dz", "s_dz", "dz"], name: bi("配对样本 t 检验", "Paired-samples t-test"),
    short: bi("对每一对的差值做单样本 t 检验，估计平均变化。", "Applies a one-sample t-test to paired differences to estimate mean change."),
    useWhen: bi("同一对象前后测量或真正一一匹配；目标是平均差值，差值分布无严重异常，且各配对之间独立。", "Use for repeated or genuinely matched observations when the estimand is the mean difference, differences have no severe anomalies, and pairs are independent."),
    h0: bi("μ_D=0，其中 D=后−前（或预先固定的相反方向）。", "μ_D=0, where D=after−before (or the prespecified reverse)."), h1: bi("μ_D≠0，或预先指定 μ_D>0 / μ_D<0。", "μ_D≠0, or a prespecified μ_D>0 / μ_D<0."),
    assumptions: bi(["每个对象恰好贡献一个差值 D，配对方向始终一致。", "不同配对之间独立。", "小样本时差值近似正态；检查差值而非两次原始测量各自的正态性。"], ["Each unit contributes one consistently oriented difference D.", "Pairs are independent of one another.", "For small samples, differences are approximately normal; assess differences, not the two raw measurements separately."]),
    formulas: [
      { label: bi("差值与 t", "Differences and t"), tex: "D_i=Y_{i,after}-Y_{i,before},\\quad t=\\frac{\\bar D}{s_D/\\sqrt n},\\quad df=n-1" },
      { label: bi("平均差置信区间", "CI for mean difference"), tex: "\\bar D\\pm t_{1-\\alpha/2,n-1}\\frac{s_D}{\\sqrt n}" },
      { label: bi("Cohen's d_z", "Cohen's d_z"), tex: "d_z=\\frac{\\bar D}{s_D}=\\frac{t}{\\sqrt n}" }
    ], symbols: [{ symbol: "D_i", meaning: bi("第 i 对按固定方向计算的差值", "difference for pair i in a fixed direction") }, { symbol: "s_D", meaning: bi("差值的样本标准差", "sample SD of differences") }, { symbol: "n", meaning: bi("完整配对数", "number of complete pairs") }],
    example: { caption: bi("治疗前后收缩压", "Systolic blood pressure before and after treatment"), columns: bi(["对象", "前", "后", "D=后−前"], ["Subject", "Before", "After", "D=after−before"]), rows: [[1,140,132,-8],[2,138,130,-8],[3,150,144,-6],[4,142,136,-6]],
      steps: bi(["差值为 −8, −8, −6, −6；D̄=−7。", "s_D=1.155，SE=1.155/√4=0.577。", "t=−7/0.577=−12.124，df=3。", "双侧 p≈.0012；95% CI≈[−8.84,−5.16]；d_z=−6.06。"], ["Differences are −8, −8, −6, −6; D̄=−7.", "s_D=1.155 and SE=1.155/√4=0.577.", "t=−7/0.577=−12.124, df=3.", "Two-sided p≈.0012; 95% CI≈[−8.84,−5.16]; d_z=−6.06."]), result: bi("平均降低约 7 mmHg；极大的 d_z 来自变化高度一致且 n 很小，应结合原始单位区间解释。", "The mean decrease is about 7 mmHg; the huge d_z reflects highly consistent changes and tiny n, so interpret the raw-unit interval too."), approximate: true },
    inference: bi("p 值检验平均差，不是分别比较前、后两列。配对可消除个体基线差异。", "The p-value tests the mean within-pair difference, not the two columns separately. Pairing removes stable between-unit variation."),
    ci: bi("示例平均变化 95% CI 约为 −8.84 到 −5.16 mmHg。", "The example 95% CI for mean change is approximately −8.84 to −5.16 mmHg."),
    effect: bi("Cohen's d_z=−6.06，符号由 D 的定义决定；d_z 是效应量，不是假设检验。", "Cohen's d_z=−6.06; its sign follows the definition of D. d_z is an effect size, not a hypothesis test."),
    edgeCases: bi(["只分析完整配对；不能把前后缺失的不同对象当作配对。", "极端差值会显著影响均值和 s_D。", "差值严重不对称且小样本时考虑精确符号检验；大致对称且秩大小可信时考虑 Wilcoxon。"], ["Analyze complete pairs; unmatched before/after observations are not paired.", "Extreme differences strongly affect the mean and s_D.", "For small, severely asymmetric differences consider an exact sign test; for roughly symmetric rankable differences consider Wilcoxon."]),
    report: bi("治疗后血压平均降低 7.0 mmHg，t(3)=−12.12，p=.0012，平均差 95% CI [−8.84,−5.16]，d_z=−6.06。", "Blood pressure decreased by 7.0 mmHg on average, t(3)=−12.12, p=.0012, 95% CI [−8.84,−5.16], d_z=−6.06."),
    python: "import numpy as np\nfrom scipy import stats\nbefore = np.array([140, 138, 150, 142])\nafter = np.array([132, 130, 144, 136])\nresult = stats.ttest_rel(after, before)\nd = after - before\nprint(result, d.mean() / d.std(ddof=1))",
    related: ["wilcoxon_signed", "exact_sign_test", "shapiro_wilk"]
  });

  add({
    id: "one_way_anova", family: "parametric", category: bi("多组比较", "Multiple-group comparison"), outcome: "continuous", design: "independent", groups: "many", kind: "comparison",
    aliases: ["anova", "f_oneway", "单因素方差分析"], name: bi("单因素 ANOVA", "One-way ANOVA"),
    short: bi("整体检验三个或更多独立总体的均值是否相同。", "Omnibus test of whether three or more independent population means are equal."),
    useWhen: bi("一个分类因素定义至少三组独立样本，结果连续，组内近似正态且方差相近。", "Use when one categorical factor defines at least three independent groups, with continuous, approximately normal outcomes and similar variances."),
    h0: bi("μ₁=μ₂=⋯=μ_k。", "μ₁=μ₂=⋯=μ_k."), h1: bi("至少一个总体均值不同。", "At least one population mean differs."),
    assumptions: bi(["对象彼此独立。", "各组残差近似正态且无严重离群值。", "总体方差相等；方差或样本量很不平衡时用 Welch ANOVA。"], ["Observations are independent.", "Within-group residuals are approximately normal without severe outliers.", "Population variances are equal; use Welch ANOVA for strong imbalance in variance or sample size."]),
    formulas: [
      { label: bi("平方和", "Sums of squares"), tex: "SS_B=\\sum_{j=1}^k n_j(\\bar x_j-\\bar x)^2,\\quad SS_W=\\sum_j\\sum_i(x_{ij}-\\bar x_j)^2" },
      { label: bi("F 统计量", "F statistic"), tex: "F=\\frac{MS_B}{MS_W}=\\frac{SS_B/(k-1)}{SS_W/(N-k)}" },
      { label: bi("效应量", "Effect size"), tex: "\\eta^2=\\frac{SS_B}{SS_T}" }
    ], symbols: [{ symbol: "k", meaning: bi("组数", "number of groups") }, { symbol: "N", meaning: bi("总样本量", "total sample size") }, { symbol: "SS_B,SS_W", meaning: bi("组间、组内平方和", "between- and within-group sums of squares") }],
    example: { caption: bi("三种方法的成绩", "Scores under three methods"), columns: bi(["组", "数值"], ["Group", "Values"]), rows: [["A","7, 8, 9"],["B","9, 10, 11"],["C","12, 13, 14"]],
      steps: bi(["组均值为 8、10、13；总均值 10.333。", "SS_B=38，SS_W=6；df_B=2，df_W=6。", "MS_B=19，MS_W=1，所以 F=19.00。", "p≈.0025；η²=38/44=.864。"], ["Group means are 8, 10, and 13; grand mean 10.333.", "SS_B=38 and SS_W=6; df_B=2 and df_W=6.", "MS_B=19 and MS_W=1, so F=19.00.", "p≈.0025; η²=38/44=.864."]), result: bi("整体差异显著；还需预先计划的对比或经校正的事后比较定位差异。", "The omnibus difference is significant; planned contrasts or multiplicity-adjusted post-hoc comparisons are still needed."), approximate: true },
    inference: bi("F 显著只说明至少一组均值不同，不说明哪两组不同。", "A significant F says at least one mean differs, not which pairs differ."),
    ci: bi("报告各组均值 CI，并为计划对比或 Tukey 事后均值差提供同时置信区间。", "Report group-mean CIs and simultaneous intervals for planned contrasts or Tukey post-hoc differences."),
    effect: bi("示例 η²=.864；小样本时 ω²通常偏差更小。", "Example η²=.864; ω² is usually less biased in small samples."),
    report: bi("三组均值存在差异，F(2,6)=19.00，p=.0025，η²=.864；随后进行经校正的组间比较。", "Group means differed, F(2,6)=19.00, p=.0025, η²=.864; multiplicity-adjusted comparisons followed."),
    python: "from scipy import stats\na, b, c = [7,8,9], [9,10,11], [12,13,14]\nprint(stats.f_oneway(a, b, c))",
    related: ["welch_anova", "kruskal_wallis", "levene_test"]
  });

  add({
    id: "welch_anova", family: "parametric", category: bi("多组比较", "Multiple-group comparison"), outcome: "continuous", design: "independent", groups: "many", kind: "comparison",
    aliases: ["welch one way", "unequal variance anova", "anova_oneway unequal"], name: bi("Welch ANOVA", "Welch ANOVA"),
    short: bi("在方差不齐时整体比较三个或更多独立总体均值。", "Compares three or more independent means without assuming equal variances."),
    useWhen: bi("多组独立连续结果，组间方差或样本量不平衡，但均值仍是目标参数。", "Use for independent continuous groups with unequal variances or sample sizes when means remain the estimand."),
    h0: bi("所有总体均值相等。", "All population means are equal."), h1: bi("至少一个总体均值不同。", "At least one population mean differs."),
    formulas: [
      { label: bi("权重与加权均值", "Weights and weighted mean"), tex: "w_j=\\frac{n_j}{s_j^2},\\quad \\bar x_w=\\frac{\\sum_j w_j\\bar x_j}{\\sum_j w_j}" },
      { label: bi("Welch F（修正形式）", "Welch F (corrected form)"), tex: "F_W=\\frac{\\sum_j w_j(\\bar x_j-\\bar x_w)^2/(k-1)}{1+\\frac{2(k-2)}{k^2-1}\\sum_j\\frac{(1-w_j/W)^2}{n_j-1}}" }
    ], symbols: [{ symbol: "w_j", meaning: bi("第 j 组的逆方差权重", "inverse-variance weight for group j") }, { symbol: "W", meaning: bi("权重总和", "sum of weights") }],
    example: { caption: bi("三组波动差异明显", "Three groups with clearly different spreads"), columns: bi(["组", "数值"], ["Group", "Values"]), rows: [["A","8, 9, 10"],["B","8, 12, 16"],["C","15, 16, 17"]],
      steps: bi(["组均值 9、12、16；方差 1、16、1。", "权重 3、0.1875、3；加权均值约 12.485。", "Welch F≈31.06，自由度约 (2, 3.62)。", "p≈.0053；整体拒绝均值相等。"], ["Means are 9, 12, and 16; variances 1, 16, and 1.", "Weights are 3, 0.1875, and 3; weighted mean ≈12.485.", "Welch F≈31.06 with df≈(2, 3.62).", "p≈.0053; reject equality of means."]), result: bi("整体显著；事后比较应使用 Games–Howell 等允许方差不齐的方法。", "The omnibus result is significant; use a heteroscedastic post-hoc method such as Games–Howell."), approximate: true },
    inference: bi("使用 Welch 修正的 F 分布与非整数分母自由度。", "Inference uses the Welch-adjusted F distribution with noninteger denominator degrees of freedom."),
    ci: bi("用 Welch/Games–Howell 标准误构造组间均值差区间，不使用 pooled 方差。", "Use Welch/Games–Howell standard errors for pairwise mean-difference intervals, not a pooled variance."),
    effect: bi("优先报告原始均值差及 CI；可补充稳健的解释方差指标。", "Prioritize raw mean differences and CIs; a robust explained-variance measure may be added."),
    report: bi("Welch ANOVA 显示均值不同，F(2,3.62)=31.06，p≈.005；使用 Games–Howell 进行事后比较。", "Welch ANOVA showed unequal means, F(2,3.62)=31.06, p≈.005; Games–Howell comparisons followed."),
    python: "from statsmodels.stats.oneway import anova_oneway\ngroups = [[8,9,10], [8,12,16], [15,16,17]]\nprint(anova_oneway(groups, use_var='unequal'))",
    references: [OFFICIAL_REFS.statsmodels, OFFICIAL_REFS.nist], related: ["one_way_anova", "kruskal_wallis", "levene_test"]
  });

  add({
    id: "repeated_anova", family: "parametric", category: bi("重复测量", "Repeated measures"), outcome: "continuous", design: "paired", groups: "many", kind: "comparison",
    aliases: ["repeated measures anova", "AnovaRM", "重复测量方差分析"], name: bi("重复测量 ANOVA", "Repeated-measures ANOVA"),
    short: bi("比较同一对象在三个或更多条件下的平均值。", "Compares means from the same units measured in three or more conditions."),
    useWhen: bi("同一对象在多个时间点或条件下连续测量，目标是平均水平差异。", "Use when the same units have continuous outcomes under multiple times or conditions and mean differences are the target."),
    h0: bi("所有条件的总体均值相等。", "All condition means are equal."), h1: bi("至少一个条件均值不同。", "At least one condition mean differs."),
    assumptions: bi(["对象之间独立，且每行对应同一对象。", "模型残差近似正态，无严重离群轨迹。", "三水平以上需考虑球形性；违反时用 Greenhouse–Geisser/Huynh–Feldt 修正或混合模型。"], ["Subjects are independent and each row belongs to one subject.", "Model residuals are approximately normal without severe trajectory outliers.", "With three or more levels, assess sphericity; use Greenhouse–Geisser/Huynh–Feldt correction or a mixed model when violated."]),
    formulas: [
      { label: bi("重复测量 F", "Repeated-measures F"), tex: "F=\\frac{MS_{condition}}{MS_{error}},\\quad df_1=k-1,\\quad df_2=(n-1)(k-1)" },
      { label: bi("偏 η²", "Partial eta squared"), tex: "\\eta_p^2=\\frac{SS_{condition}}{SS_{condition}+SS_{error}}" }
    ], symbols: [{ symbol: "n", meaning: bi("完整对象数", "number of complete subjects") }, { symbol: "k", meaning: bi("重复条件数", "number of repeated conditions") }],
    example: { caption: bi("5 名对象的三次评分", "Three scores from five subjects"), columns: bi(["对象", "T1", "T2", "T3"], ["Subject", "T1", "T2", "T3"]), rows: [[1,10,9,9],[2,12,11,9],[3,11,10,8],[4,13,11,10],[5,12,10,9]],
      steps: bi(["时间均值为 11.6、10.2、9.0。", "SS_time=16.933，SS_error=2.400。", "F=(16.933/2)/(2.400/8)=28.22。", "df=(2,8)，p≈.00024；partial η²≈.876。"], ["Time means are 11.6, 10.2, and 9.0.", "SS_time=16.933 and SS_error=2.400.", "F=(16.933/2)/(2.400/8)=28.22.", "df=(2,8), p≈.00024; partial η²≈.876."]), result: bi("时间总体效应显著；仍需经校正的计划对比来定位变化。", "The omnibus time effect is significant; multiplicity-adjusted planned contrasts are needed to localize change."), approximate: true },
    inference: bi("若球形性不满足，报告校正后的自由度和 p 值，而非未校正结果。", "If sphericity fails, report corrected degrees of freedom and p-value rather than the uncorrected result."),
    ci: bi("报告各时间点估计均值与重复测量对比的 CI；对比区间需处理多重性。", "Report estimated means and CIs for repeated-measures contrasts with multiplicity control."),
    effect: bi("示例 partial η²≈.876；也可报告 generalized η²。", "Example partial η²≈.876; generalized η² is another option."),
    report: bi("时间主效应显著，F(2,8)=28.22，p<.001，partial η²=.876；报告球形性检查及校正。", "The time effect was significant, F(2,8)=28.22, p<.001, partial η²=.876; sphericity diagnostics and any correction were reported."),
    python: "import pandas as pd\nfrom statsmodels.stats.anova import AnovaRM\nwide = [[10,9,9],[12,11,9],[11,10,8],[13,11,10],[12,10,9]]\ndf = pd.DataFrame([(i,t,v) for i,row in enumerate(wide) for t,v in enumerate(row)], columns=['subject','time','score'])\nprint(AnovaRM(df, 'score', 'subject', within=['time']).fit())",
    references: [OFFICIAL_REFS.statsmodels, OFFICIAL_REFS.nist], related: ["friedman", "paired_t"]
  });

  add({
    id: "mann_whitney", family: "nonparametric", category: bi("秩检验", "Rank tests"), outcome: "ordinal", design: "independent", groups: "two", kind: "comparison",
    aliases: ["mann-whitney u", "wilcoxon rank sum", "ranksums", "秩和检验"], name: bi("Mann–Whitney U 检验", "Mann–Whitney U test"),
    short: bi("用秩比较两个独立分布；并非自动等同于“中位数检验”。", "Uses ranks to compare two independent distributions; it is not automatically a test of medians."),
    useWhen: bi("两组独立、有序或连续结果，数值可可靠排序；关注随机抽取的一组值大于另一组值的倾向。", "Use for two independent ordinal or continuous samples when values can be meaningfully ranked and stochastic ordering is of interest."),
    h0: bi("两组观测的成对优胜概率为 0.5；在完整同分布原假设下，两组分布相同。", "The probability that a random observation from one group exceeds one from the other is 0.5; under the full null, distributions are identical."),
    h1: bi("两组的秩分布或优胜概率不同。", "Rank distributions or the probability of superiority differ."),
    assumptions: bi(["两组及组内观测独立。", "结果至少有序；并列值使用平均秩和校正。", "只有分布形状相近时，位置差异才可简化解释为中位数差异。"], ["Groups and within-group observations are independent.", "The outcome is at least ordinal; ties use average ranks and correction.", "Only with similarly shaped distributions can a location shift be simplified to a median shift."]),
    formulas: [
      { label: bi("U 统计量", "U statistic"), tex: "U_1=R_1-\\frac{n_1(n_1+1)}2,\\quad U_2=n_1n_2-U_1" },
      { label: bi("秩二列相关", "Rank-biserial correlation"), tex: "r_{rb}=\\frac{2U_1}{n_1n_2}-1" }
    ], symbols: [{ symbol: "R_1", meaning: bi("第 1 组秩和", "rank sum for group 1") }, { symbol: "U_1", meaning: bi("按第 1 组方向定义的 U", "U oriented to group 1") }],
    example: { caption: bi("两个互不重叠的小样本", "Two nonoverlapping small samples"), columns: bi(["组", "数值"], ["Group", "Values"]), rows: [["A","1, 2, 3, 4, 5"],["B","6, 7, 8, 9, 10"]],
      steps: bi(["合并排序后 A 的秩为 1–5，R_A=15。", "U_A=15−5×6/2=0。", "无并列时枚举所有分组，双侧精确 p=2/252≈.00794。", "r_rb=2×0/25−1=−1。"], ["After pooled ranking, A receives ranks 1–5, so R_A=15.", "U_A=15−5×6/2=0.", "With no ties, enumeration gives two-sided exact p=2/252≈.00794.", "r_rb=2×0/25−1=−1."]), result: bi("A 的所有观测都小于 B；方向性效应达到边界 −1。", "Every A observation is below every B observation; the oriented effect reaches −1."), approximate: false },
    inference: bi("小样本无并列时可用精确分布；大样本或并列值常用含并列校正的正态近似。", "Use the exact distribution for small tie-free samples; large samples or ties often use a tie-corrected normal approximation."),
    ci: bi("可报告 Hodges–Lehmann 位置差及分布无关 CI，但只有位移模型成立时才解释为中位数差。", "A Hodges–Lehmann shift and distribution-free CI may be reported, but interpret it as a median difference only under a shift model."),
    effect: bi("示例 r_rb=−1；也可报告概率优势或 Cliff's delta，并说明方向。", "Example r_rb=−1; probability of superiority or Cliff's delta may also be reported with direction."),
    report: bi("A 组秩低于 B 组，Mann–Whitney U=0，双侧精确 p=.0079，r_rb=−1.00。", "Group A had lower ranks than B, Mann–Whitney U=0, two-sided exact p=.0079, r_rb=−1.00."),
    python: "from scipy import stats\na = [1,2,3,4,5]\nb = [6,7,8,9,10]\nprint(stats.mannwhitneyu(a, b, alternative='two-sided', method='exact'))",
    related: ["independent_t", "welch_t", "kruskal_wallis"]
  });

  add({
    id: "wilcoxon_signed", family: "nonparametric", category: bi("配对秩检验", "Paired rank tests"), outcome: "ordinal", design: "paired", groups: "two", kind: "comparison",
    aliases: ["paired_wilcoxon", "wilcoxon_signed_rank", "wilcoxon paired", "符号秩检验"], name: bi("Wilcoxon 符号秩检验", "Wilcoxon signed-rank test"),
    short: bi("利用非零配对差值的方向和绝对值秩来检验对称位置变化。", "Uses signs and absolute ranks of nonzero paired differences to test a symmetric location shift."),
    useWhen: bi("配对差值至少可排序、分布大致对称，均值模型不合适但差值大小仍有意义。", "Use when paired differences are rankable and roughly symmetric, a mean model is unsuitable, but magnitudes remain informative."),
    h0: bi("差值分布关于 0 对称（等价地，在位移模型下伪中位数为 0）。", "The difference distribution is symmetric about zero (equivalently, pseudomedian zero under a shift model)."),
    h1: bi("对称中心不为 0，或预先指定方向。", "The symmetry center is nonzero, or a prespecified direction."),
    assumptions: bi(["使用固定方向的配对差值，配对之间独立。", "绝对差值可可靠排序；并列秩取平均。", "把检验解释为位置/中位数变化时要求差值分布大致对称。"], ["Use consistently oriented paired differences; pairs are independent.", "Absolute differences can be ranked; ties receive average ranks.", "Interpreting the test as a location or median shift requires roughly symmetric differences."]),
    formulas: [
      { label: bi("符号秩和", "Signed-rank sums"), tex: "W^+=\\sum_{D_i>0}R_i,\\quad W^-=\\sum_{D_i<0}R_i,\\quad R_i=rank(|D_i|)" },
      { label: bi("双侧统计量与效应量", "Two-sided statistic and effect"), tex: "W=\\min(W^+,W^-),\\quad r_{rb}=\\frac{W^+-W^-}{W^++W^-}" }
    ], symbols: [{ symbol: "D_i", meaning: bi("非零配对差值", "nonzero paired difference") }, { symbol: "R_i", meaning: bi("绝对差值的平均秩", "average rank of the absolute difference") }],
    example: { caption: bi("6 个非零配对差值", "Six nonzero paired differences"), columns: bi(["对象", "D=后−前"], ["Subject", "D=after−before"]), rows: [[1,-2],[2,-3],[3,-1],[4,-4],[5,-2],[6,-5]],
      steps: bi(["绝对差值为 2,3,1,4,2,5；两个 2 各取平均秩 2.5。", "所有差值为负，所以 W⁺=0，W⁻=21。", "双侧统计量 W=0；无零差时精确 p=2/2⁶=.03125。", "r_rb=(0−21)/21=−1。"], ["Absolute differences are 2,3,1,4,2,5; tied 2s each receive rank 2.5.", "All differences are negative, so W⁺=0 and W⁻=21.", "The two-sided statistic is W=0; with no zero differences exact p=2/2⁶=.03125.", "r_rb=(0−21)/21=−1."]), result: bi("差值一致为负。不同软件对并列、零差和“精确”算法的定义可能不同。", "All differences are negative. Software definitions of exact handling can differ with ties and zeros."), approximate: false },
    inference: bi("精确枚举假定适当处理零差与并列；使用软件时报告 zero_method、精确或近似算法。", "Exact enumeration depends on zero and tie handling; report the software's zero_method and exact/approximate algorithm."),
    ci: bi("可报告 Walsh 平均数的 Hodges–Lehmann 伪中位数及其区间；并列/零差会影响可达到的置信水平。", "Report a Hodges–Lehmann pseudomedian based on Walsh averages with its interval; ties and zeros affect attainable confidence levels."),
    effect: bi("示例秩二列相关 r_rb=−1；常见 z/√n 效应量依赖正态近似。", "Example rank-biserial correlation r_rb=−1; the common z/√n effect depends on a normal approximation."),
    edgeCases: bi(["零差不参与方向信息；Pratt 与 Wilcox 处理规则不同。", "并列绝对差值使用平均秩，可能使简单精确分布不再适用。", "差值严重不对称或只信任方向时用精确符号检验。"], ["Zero differences carry no directional information; Pratt and Wilcox conventions differ.", "Tied absolute differences use average ranks and may invalidate a simple exact distribution.", "Use an exact sign test for severely asymmetric differences or when only direction is trusted."]),
    report: bi("Wilcoxon 符号秩检验显示差值为负，W=0，双侧精确 p=.0313，r_rb=−1.00；报告了零差和并列处理。", "The Wilcoxon signed-rank test indicated negative differences, W=0, two-sided exact p=.0313, r_rb=−1.00; zero and tie handling were reported."),
    python: "from scipy import stats\nd = [-2,-3,-1,-4,-2,-5]\nprint(stats.wilcoxon(d, alternative='two-sided', method='auto', zero_method='wilcox'))",
    related: ["paired_t", "exact_sign_test", "friedman"]
  });

  add({
    id: "exact_sign_test", family: "nonparametric", category: bi("配对方向检验", "Paired direction tests"), outcome: "ordinal", design: "paired", groups: "two", kind: "comparison",
    aliases: ["paired_sign_test", "binomial sign test", "sign test", "精确符号检验"], name: bi("精确符号检验", "Exact sign test"),
    short: bi("忽略零差和差值大小，只检验非 tie 配对中正、负方向是否偏离 50%。", "Discards zero differences and magnitudes, testing whether positive versus negative directions among non-ties depart from 50%."),
    useWhen: bi("配对数据只能可靠判断方向，差值严重不对称、含极端值，或量尺不支持比较差值大小。", "Use when only direction is trustworthy, paired differences are severely asymmetric or extreme, or the scale does not support comparing magnitudes."),
    h0: bi("P(D>0 | D≠0)=0.5。", "P(D>0 | D≠0)=0.5."), h1: bi("该条件概率不等于 0.5，或预先指定大于/小于 0.5。", "This conditional probability differs from 0.5, or is prespecified as greater/less than 0.5."),
    assumptions: bi(["配对方向定义一致，不同配对之间独立。", "零差（tie）从二项试验的有效样本量中删除，并单独报告。", "连续差值时可联系中位数；有质量点或大量 ties 时应直接解释为条件方向概率。"], ["Difference direction is consistent and pairs are independent.", "Zero differences (ties) are removed from the binomial effective sample and reported separately.", "For continuous differences it connects to a median; with point masses or many ties, interpret the conditional directional probability directly."]),
    formulas: [
      { label: bi("有效样本与正向次数", "Effective sample and positive count"), tex: "n_{eff}=n_++n_-,\\quad X=n_+\\sim Binomial(n_{eff},0.5)\\quad(H_0)" },
      { label: bi("双侧精确 p 值", "Two-sided exact p-value"), tex: "p=\\sum_{x:\\,P(X=x)\\le P(X=n_+)}P(X=x)\\;\\;\\text{(probability ordering)}" },
      { label: bi("简单方向效应", "Simple directional effect"), tex: "\\Delta_{sign}=\\frac{n_+-n_-}{n_{eff}}=2\\hat p_+-1" }
    ], symbols: [{ symbol: "n_+,n_-", meaning: bi("正、负非零差值数", "numbers of positive and negative nonzero differences") }, { symbol: "n_{eff}", meaning: bi("删除 ties 后的有效配对数", "effective pairs after removing ties") }],
    example: { caption: bi("10 个非 tie 配对中 9 个正向", "Nine positives among ten non-tied pairs"), columns: bi(["方向", "次数"], ["Direction", "Count"]), rows: [["+",9],["−",1],["tie",2]],
      steps: bi(["删除 2 个 tie，n_eff=10；n₊=9。", "H₀ 下 X~Binomial(10,.5)。", "双侧 p=2[P(X≤1)]=2(1+10)/2¹⁰=.021484。", "正向概率估计 .90；Clopper–Pearson 95% CI 约 [.555,.997]。"], ["Remove 2 ties, giving n_eff=10 and n₊=9.", "Under H₀, X~Binomial(10,.5).", "Two-sided p=2[P(X≤1)]=2(1+10)/2¹⁰=.021484.", "Estimated positive probability is .90; Clopper–Pearson 95% CI ≈[.555,.997]."]), result: bi("非 tie 配对中正向次数显著多于反向；Δ_sign=.80。", "Positive directions significantly outnumber negative directions among non-ties; Δ_sign=.80."), approximate: true },
    inference: bi("“精确”指直接用二项分布，不用正态近似。双侧定义在离散分布中可能略有差别，应说明软件规则。", "Exact means using the binomial distribution directly, not a normal approximation. Two-sided conventions can differ for discrete distributions, so state the software rule."),
    ci: bi("对正向条件概率报告 Clopper–Pearson 区间；对连续差值中位数可用次序统计量区间。", "Report a Clopper–Pearson interval for the conditional positive probability; for a continuous-difference median, an order-statistic interval is possible."),
    effect: bi("Δ_sign=.80，表示非 tie 配对中净正向比例为 80 个百分点。", "Δ_sign=.80, a net positive directional proportion of 80 percentage points among non-ties."),
    edgeCases: bi(["ties 不应随意分配正负；同时报告原始配对数、tie 数和 n_eff。", "检验完全忽略差值大小，因此通常比 Wilcoxon 或配对 t 功效低。", "它不是 Fisher 精确检验。"], ["Do not arbitrarily assign ties; report total pairs, ties, and n_eff.", "The test ignores magnitudes and is usually less powerful than Wilcoxon or paired t when those models fit.", "It is not Fisher's exact test."]),
    report: bi("删除 2 个零差后，10 个有效配对中 9 个为正；精确符号检验 p=.0215，正向概率=.90，95% CI [.555,.997]，Δ_sign=.80。", "After removing 2 zero differences, 9 of 10 effective pairs were positive; exact sign test p=.0215, positive probability=.90, 95% CI [.555,.997], Δ_sign=.80."),
    python: "from scipy import stats\npositive, negative, ties = 9, 1, 2\nresult = stats.binomtest(positive, positive + negative, p=0.5, alternative='two-sided')\nprint(result.pvalue, result.proportion_ci(method='exact'))",
    related: ["wilcoxon_signed", "paired_t", "mcnemar"]
  });

  add({
    id: "kruskal_wallis", family: "nonparametric", category: bi("秩检验", "Rank tests"), outcome: "ordinal", design: "independent", groups: "many", kind: "comparison",
    aliases: ["kruskal", "kruskal wallis h", "非参数单因素"], name: bi("Kruskal–Wallis 检验", "Kruskal–Wallis test"),
    short: bi("用合并秩整体比较三个或更多独立组。", "Uses pooled ranks for an omnibus comparison of three or more independent groups."),
    useWhen: bi("至少三组独立、有序或非正态连续结果，关注分布位置/随机优胜。", "Use for at least three independent ordinal or non-normal continuous groups when rank location or stochastic ordering is of interest."),
    h0: bi("各组秩分布相同；完整原假设为各组分布相同。", "Group rank distributions are identical; the full null is equal distributions."), h1: bi("至少一组秩分布不同。", "At least one group's rank distribution differs."),
    assumptions: bi(["各组和组内观测独立。", "结果至少有序；并列使用平均秩及校正。", "只有形状相近时才把差异简化为中位数/位置差异。"], ["Groups and observations are independent.", "Outcomes are at least ordinal; ties use average ranks and correction.", "Only with similar shapes should differences be simplified to medians or locations."]),
    formulas: [
      { label: bi("未校正 H", "Uncorrected H"), tex: "H=\\frac{12}{N(N+1)}\\sum_{j=1}^k\\frac{R_j^2}{n_j}-3(N+1)" },
      { label: bi("并列校正", "Tie correction"), tex: "H_c=\\frac{H}{1-\\sum_g(t_g^3-t_g)/(N^3-N)}" },
      { label: bi("效应量", "Effect size"), tex: "\\epsilon^2=\\frac{H_c-k+1}{N-k}" }
    ], symbols: [{ symbol: "R_j", meaning: bi("第 j 组秩和", "rank sum for group j") }, { symbol: "t_g", meaning: bi("第 g 个并列组大小", "size of tie group g") }],
    example: { caption: bi("三组有序评分", "Three groups of ordinal scores"), columns: bi(["组", "评分"], ["Group", "Scores"]), rows: [["A","2, 3, 3"],["B","4, 4, 5"],["C","6, 7, 7"]],
      steps: bi(["合并平均秩后，各组秩和为 6、15、24。", "未校正 H=7.20；并列修正因子=.975。", "H_c=7.385，df=2，渐近 p≈.0249。", "ε²=(7.385−3+1)/(9−3)=.897。"], ["Pooled average ranks give rank sums 6, 15, and 24.", "Uncorrected H=7.20; tie correction factor=.975.", "H_c=7.385, df=2, asymptotic p≈.0249.", "ε²=(7.385−3+1)/(9−3)=.897."]), result: bi("整体秩差异显著；样本很小，渐近 p 值需谨慎，可考虑置换检验。", "The omnibus rank difference is significant; with tiny samples, treat the asymptotic p-value cautiously and consider permutation inference."), approximate: true },
    inference: bi("常用 χ²_(k−1) 近似；显著后用 Dunn/Conover 等经多重校正的比较。", "The usual reference is χ²_(k−1); follow a significant result with multiplicity-adjusted Dunn/Conover comparisons."),
    ci: bi("为经校正的两两位置差、概率优势或中位数分别报告区间；不要只给整体 H。", "Report adjusted intervals for pairwise shifts, probability of superiority, or medians rather than H alone."),
    effect: bi("示例 ε²≈.897；小样本估计可能不稳定。", "Example ε²≈.897; small-sample estimates can be unstable."),
    report: bi("三组秩分布不同，Kruskal–Wallis H(2)=7.385，渐近 p=.025，ε²=.897；随后进行校正的 Dunn 比较。", "Rank distributions differed, Kruskal–Wallis H(2)=7.385, asymptotic p=.025, ε²=.897; adjusted Dunn comparisons followed."),
    python: "from scipy import stats\na, b, c = [2,3,3], [4,4,5], [6,7,7]\nprint(stats.kruskal(a, b, c))",
    related: ["one_way_anova", "welch_anova", "mann_whitney"]
  });

  add({
    id: "friedman", family: "nonparametric", category: bi("重复测量秩检验", "Repeated rank tests"), outcome: "ordinal", design: "paired", groups: "many", kind: "comparison",
    aliases: ["friedman test", "friedmanchisquare", "非参数重复测量"], name: bi("Friedman 检验", "Friedman test"),
    short: bi("在每个对象内部排序，比较三个或更多重复条件。", "Ranks conditions within each subject to compare three or more repeated conditions."),
    useWhen: bi("同一对象在至少三个条件下产生有序或非正态连续结果。", "Use for ordinal or non-normal continuous outcomes from the same subjects in at least three conditions."),
    h0: bi("各条件的秩分布/位置相同。", "Condition rank distributions or locations are equal."), h1: bi("至少一个条件不同。", "At least one condition differs."),
    assumptions: bi(["对象之间独立，每个对象在所有条件均有测量。", "在对象内部数值可排序，并列取平均秩。", "若有缺失或复杂协变量，考虑混合模型或适当置换方法。"], ["Subjects are independent and measured in every condition.", "Values are rankable within subject; ties receive average ranks.", "For missing data or covariates, consider a mixed model or appropriate permutation method."]),
    formulas: [
      { label: bi("Friedman Q", "Friedman Q"), tex: "Q=\\frac{12}{nk(k+1)}\\sum_{j=1}^kR_j^2-3n(k+1)" },
      { label: bi("Kendall's W", "Kendall's W"), tex: "W=\\frac{Q}{n(k-1)}" }
    ], symbols: [{ symbol: "R_j", meaning: bi("条件 j 的对象内秩和", "sum of within-subject ranks for condition j") }, { symbol: "n,k", meaning: bi("对象数、条件数", "numbers of subjects and conditions") }],
    example: { caption: bi("与重复测量 ANOVA 相同的 5×3 数据", "The same 5×3 data used for repeated-measures ANOVA"), columns: bi(["对象", "T1", "T2", "T3"], ["Subject", "T1", "T2", "T3"]), rows: [[1,10,9,9],[2,12,11,9],[3,11,10,8],[4,13,11,10],[5,12,10,9]],
      steps: bi(["逐行排序；第一行 T2=T3，各取秩 1.5。", "条件秩和为 15、9.5、5.5。", "并列校正后 Q≈9.579，df=2，渐近 p≈.0083。", "Kendall's W=9.579/[5×2]=.958。"], ["Rank within rows; in row 1 T2=T3 and each gets rank 1.5.", "Condition rank sums are 15, 9.5, and 5.5.", "Tie-corrected Q≈9.579, df=2, asymptotic p≈.0083.", "Kendall's W=9.579/[5×2]=.958."]), result: bi("条件总体差异显著；随后进行经校正的成对 Wilcoxon 比较。", "The omnibus condition difference is significant; follow with adjusted paired Wilcoxon comparisons."), approximate: true },
    inference: bi("Q 通常与 χ²_(k−1) 比较；极小样本可用精确或置换分布。", "Q is usually compared with χ²_(k−1); exact or permutation distributions are preferable for very small samples."),
    ci: bi("整体检验本身没有单一差值 CI；对预先指定的成对变化报告 Hodges–Lehmann 区间。", "The omnibus test has no single difference CI; report Hodges–Lehmann intervals for prespecified pairwise changes."),
    effect: bi("示例 Kendall's W≈.958，0 表示无一致条件效应，1 表示完全一致排序。", "Example Kendall's W≈.958; 0 indicates no consistent condition effect and 1 perfect agreement."),
    report: bi("条件评分不同，Friedman Q(2)=9.579，渐近 p=.008，Kendall's W=.958。", "Condition scores differed, Friedman Q(2)=9.579, asymptotic p=.008, Kendall's W=.958."),
    python: "from scipy import stats\nt1 = [10,12,11,13,12]\nt2 = [9,11,10,11,10]\nt3 = [9,9,8,10,9]\nprint(stats.friedmanchisquare(t1, t2, t3))",
    related: ["repeated_anova", "wilcoxon_signed"]
  });

  add({
    id: "one_proportion_z", family: "categorical", category: bi("比例比较", "Proportion comparison"), outcome: "categorical", design: "one", groups: "one", kind: "comparison",
    aliases: ["z test", "ztest", "z-test", "z 检验", "z检验", "proportion z test", "one proportion z", "one-sample proportion z-test", "one sample proportion z", "1 sample proportion", "proportions_ztest", "statsmodels proportions_ztest", "单样本比例z检验", "单比例z检验"],
    name: bi("单样本比例 Z 检验", "One-sample proportion Z-test"),
    short: bi("用正态近似检验一个二项总体比例是否等于指定值。", "Uses a normal approximation to test whether one binomial proportion equals a specified value."),
    background: bi("在 H₀:p=p₀ 下，用 p₀ 决定样本比例的标准误，再把 p̂−p₀ 标准化为 Z。", "Under H₀:p=p₀, it uses p₀ to determine the standard error of the sample proportion and standardizes p̂−p₀."),
    useWhen: bi("独立对象各产生一次成功/失败结果，p₀ 预先给定，且 np₀ 与 n(1−p₀) 足够大，使 H₀ 下正态近似可靠。", "Use for independent Bernoulli outcomes with a prespecified p₀ when np₀ and n(1−p₀) are large enough for the null normal approximation."),
    avoidWhen: bi("H₀ 下成功或失败期望数很小、p₀ 接近 0/1、对象聚类/重复或使用复杂抽样权重时。小样本优先精确二项检验。", "Avoid it when expected successes or failures under H₀ are small, p₀ is near 0/1, units are clustered/repeated, or survey weights are complex. Prefer an exact binomial test for small samples."),
    h0: bi("p=p₀。", "p=p₀."), h1: bi("p≠p₀（双侧），或预先指定 p>p₀ / p<p₀。", "p≠p₀ (two-sided), or a prespecified p>p₀ / p<p₀."),
    assumptions: bi([
      "n 个 Bernoulli 结果独立且成功定义一致；x 是成功数。",
      "p₀ 在看数据前给定，且不是由同一样本估计。",
      "H₀ 下 np₀ 与 n(1−p₀) 通常都至少约 5–10；这只是经验线，近似可疑时应使用精确二项法。"
    ], [
      "The n Bernoulli outcomes are independent with a consistent success definition; x is the success count.",
      "p₀ is prespecified and not estimated from the same sample.",
      "Under H₀, np₀ and n(1−p₀) are usually at least about 5–10; this is only a rule of thumb, so use exact binomial inference when the approximation is doubtful."
    ]),
    formulas: [
      { label: bi("样本比例与 H₀ 标准误", "Sample proportion and null standard error"), tex: "\\hat p=\\frac{x}{n},\\qquad SE_0=\\sqrt{\\frac{p_0(1-p_0)}{n}}" },
      { label: bi("Z 与双侧 p", "Z and two-sided p"), tex: "Z=\\frac{\\hat p-p_0}{SE_0},\\qquad p_{two}=2\\{1-\\Phi(|z_{obs}|)\\}" },
      { label: bi("Wilson 区间", "Wilson interval"), tex: "C=\\frac{\\hat p+z_*^2/(2n)}{1+z_*^2/n},\\quad H=\\frac{z_*}{1+z_*^2/n}\\sqrt{\\frac{\\hat p(1-\\hat p)}n+\\frac{z_*^2}{4n^2}},\\quad CI=[C-H,C+H]" },
      { label: bi("Cohen's h", "Cohen's h"), tex: "h=2\\arcsin\\sqrt{\\hat p}-2\\arcsin\\sqrt{p_0}" }
    ],
    symbols: [
      { symbol: "x,n", meaning: bi("成功数与独立试验数", "successes and independent trials") },
      { symbol: "\\hat p", meaning: bi("样本成功比例", "sample success proportion") },
      { symbol: "p_0", meaning: bi("原假设比例", "null proportion") },
      { symbol: "\\Phi", meaning: bi("标准正态累计分布函数", "standard normal CDF") },
      { symbol: "z_*", meaning: bi("置信水平对应的标准正态分位数", "standard-normal critical value") }
    ],
    example: {
      caption: bi("200 人中 118 人成功，与 50% 比较", "118 successes among 200 people compared with 50%"),
      columns: bi(["结果", "原始计数"], ["Outcome", "Raw count"]), rows: [["成功 / Success",118],["失败 / Failure",82]],
      steps: bi([
        "x=118，n=200，p̂=118/200=.590，p₀=.500。",
        "SE₀=√[.5×.5/200]=.035355。",
        "Z=(.590−.500)/.035355=2.545584。",
        "双侧正态近似 p=.010909。",
        "Wilson 95% CI=[.520765,.655843]；比例差=.090；Cohen's h=.180986。"
      ], [
        "x=118, n=200, p̂=.590, and p₀=.500.",
        "SE₀=√[.5×.5/200]=.035355.",
        "Z=(.590−.500)/.035355=2.545584.",
        "Two-sided normal-approximation p=.010909.",
        "Wilson 95% CI=[.520765,.655843]; proportion difference=.090; Cohen's h=.180986."
      ]),
      result: bi("成功比例高于 50%；Wilson 区间完整高于 .5，但证据与效应大小仍应分别解释。", "The success proportion is above 50%; the Wilson interval lies above .5, while evidence and effect magnitude should still be interpreted separately."),
      approximate: true
    },
    inference: bi("这里的 p 是 H₀ 方差 p₀(1−p₀)/n 下的正态近似尾面积。小样本用精确二项检验，并说明双侧概率规则。", "The p-value is a normal-approximation tail area using the null variance p₀(1−p₀)/n. For small samples use an exact binomial test and state the two-sided convention."),
    ci: bi("示例 Wilson 95% CI=[.5208,.6558]；Wilson 通常比 Wald p̂±1.96√[p̂(1−p̂)/n] 更稳健。", "The example Wilson 95% CI is [.5208,.6558]; Wilson is generally more reliable than the Wald interval p̂±1.96√[p̂(1−p̂)/n]."),
    effect: bi("原始比例差 p̂−p₀=.090（9 个百分点）；Cohen's h=.181。优先用百分点和区间解释。", "The raw difference p̂−p₀=.090 (9 percentage points), with Cohen's h=.181. Prefer percentage points and their interval for interpretation."),
    edgeCases: bi([
      "x=0 或 x=n 时 Wald 标准误和区间尤其失真；使用 Wilson、Jeffreys 或精确区间。",
      "同一对象重复测量、家庭或中心聚类会使有效样本量小于 n。",
      "非概率抽样时，检验只量化模型内抽样误差，不能修复选择偏差。"
    ], [
      "When x=0 or x=n, Wald SEs and intervals are especially misleading; use Wilson, Jeffreys, or exact intervals.",
      "Repeated outcomes or household/site clustering reduce the effective sample size below n.",
      "With nonprobability sampling, the test quantifies model-based sampling error and cannot repair selection bias."
    ]),
    report: bi("200 人中 118 人成功（59.0%）；单样本比例 Z=2.546，双侧 p=.0109，比例差=9.0 个百分点，Wilson 95% CI [.5208,.6558]，h=.181。", "Among 200 people, 118 succeeded (59.0%); one-sample proportion Z=2.546, two-sided p=.0109, difference=9.0 percentage points, Wilson 95% CI [.5208,.6558], h=.181."),
    python: "from statsmodels.stats.proportion import proportions_ztest, proportion_confint\ncount, nobs, p0 = 118, 200, 0.5\nz, p = proportions_ztest(count, nobs, value=p0, alternative='two-sided', prop_var=p0)\nci = proportion_confint(count, nobs, alpha=0.05, method='wilson')\nprint(z, p, ci)\n# Small-sample exact fallback:\n# from scipy import stats\n# print(stats.binomtest(count, nobs, p=p0))",
    references: [OFFICIAL_REFS.proportionsZ, OFFICIAL_REFS.scipyBinom, OFFICIAL_REFS.nist],
    related: ["two_proportion_z", "chi_square_goodness", "logistic_regression"]
  });

  add({
    id: "two_proportion_z", family: "categorical", category: bi("比例比较", "Proportion comparison"), outcome: "categorical", design: "independent", groups: "two", kind: "comparison",
    aliases: ["z test", "ztest", "z-test", "z 检验", "z检验", "proportion z test", "two proportion z", "two-proportion z-test", "two-sample proportions z-test", "two sample proportion z", "2 sample proportion", "pooled proportion z", "difference in proportions z test", "proportions_ztest", "statsmodels proportions_ztest", "两样本比例z检验", "两独立样本比例z检验", "两比例z检验"],
    name: bi("两独立样本比例 Z 检验", "Two-sample proportions Z-test"),
    short: bi("用合并的 H₀ 方差比较两个独立二项总体的成功比例。", "Compares two independent binomial proportions using the pooled null variance."),
    background: bi("H₀:p₁=p₂ 时先把两组合并估计共同概率，再用该概率计算比例差的标准误。", "Under H₀:p₁=p₂, it first pools both groups to estimate their common probability, then uses it for the standard error of the difference."),
    useWhen: bi("两个相互独立样本各产生成功/失败计数，目标是比例差，且 H₀ 下每组期望成功和失败数足以支持正态近似。", "Use for success/failure counts from two independent samples when the target is a proportion difference and expected successes and failures under H₀ support a normal approximation."),
    avoidWhen: bi("两组是同一对象前后或匹配样本（改用 McNemar）；任一格计数很小（考虑 Fisher 精确检验）；存在聚类、重复或复杂权重而未调整。", "Do not use for paired or matched samples (use McNemar), very small cells (consider Fisher's exact test), or unadjusted clustering, repeats, or complex weights."),
    h0: bi("p₁−p₂=0。", "p₁−p₂=0."), h1: bi("p₁−p₂≠0，或预先指定方向。", "p₁−p₂≠0, or a prespecified direction."),
    assumptions: bi([
      "两样本相互独立，且各样本内对象独立；成功定义一致。",
      "输入每组原始成功数 xᵢ 和总数 nᵢ，而非只输入百分比。",
      "H₀ 下 nᵢp_pool 与 nᵢ(1−p_pool) 足够大；稀疏表使用精确方法。"
    ], [
      "Samples are independent of each other and units are independent within samples; success is defined consistently.",
      "Provide raw successes xᵢ and totals nᵢ, not percentages alone.",
      "Under H₀, nᵢp_pool and nᵢ(1−p_pool) are sufficiently large; use exact methods for sparse tables."
    ]),
    formulas: [
      { label: bi("组比例与 H₀ 合并比例", "Group and pooled null proportions"), tex: "\\hat p_i=\\frac{x_i}{n_i},\\qquad \\hat p_{pool}=\\frac{x_1+x_2}{n_1+n_2}" },
      { label: bi("比例相等的 Z 检验", "Z-test for equal proportions"), tex: "Z=\\frac{\\hat p_1-\\hat p_2}{\\sqrt{\\hat p_{pool}(1-\\hat p_{pool})(1/n_1+1/n_2)}}" },
      { label: bi("双侧 p 值", "Two-sided p-value"), tex: "p_{two}=2\\{1-\\Phi(|z_{obs}|)\\}" },
      { label: bi("未合并 Wald 差值区间（基础公式）", "Unpooled Wald difference interval (basic formula)"), tex: "(\\hat p_1-\\hat p_2)\\pm z_*\\sqrt{\\frac{\\hat p_1(1-\\hat p_1)}{n_1}+\\frac{\\hat p_2(1-\\hat p_2)}{n_2}}" },
      { label: bi("推荐的 Newcombe–Wilson 组合", "Recommended Newcombe–Wilson combination"), tex: "L=d-\\sqrt{(\\hat p_1-L_1)^2+(U_2-\\hat p_2)^2},\\quad U=d+\\sqrt{(U_1-\\hat p_1)^2+(\\hat p_2-L_2)^2}" },
      { label: bi("Cohen's h", "Cohen's h"), tex: "h=2\\arcsin\\sqrt{\\hat p_1}-2\\arcsin\\sqrt{\\hat p_2}" }
    ],
    symbols: [
      { symbol: "x_i,n_i", meaning: bi("第 i 组成功数与总数", "successes and total in group i") },
      { symbol: "\\hat p_{pool}", meaning: bi("H₀:p₁=p₂ 下的共同概率估计", "common probability estimate under H₀:p₁=p₂") },
      { symbol: "d", meaning: bi("观察风险差 p̂₁−p̂₂", "observed risk difference p̂₁−p̂₂") },
      { symbol: "[L_i,U_i]", meaning: bi("第 i 组 Wilson 比例区间", "Wilson interval for group i") }
    ],
    example: {
      caption: bi("A、B 两组成功比例", "Success proportions in groups A and B"),
      columns: bi(["组", "成功", "失败", "总数", "比例"], ["Group", "Success", "Failure", "Total", "Proportion"]),
      rows: [["A",110,90,200,".550"],["B",82,118,200,".410"]],
      steps: bi([
        "p̂₁=.550，p̂₂=.410，风险差 d=.140。",
        "H₀ 合并比例 p_pool=(110+82)/400=.480。",
        "SE₀=√[.48×.52×(1/200+1/200)]=.049960。",
        "Z=.140/.049960=2.802243；双侧 p=.005075。",
        "Newcombe–Wilson 95% CI for d=[.042080,.234195]；RR=1.341，OR=1.759，h=.281154。"
      ], [
        "p̂₁=.550, p̂₂=.410, and risk difference d=.140.",
        "The pooled null proportion is p_pool=(110+82)/400=.480.",
        "SE₀=√[.48×.52×(1/200+1/200)]=.049960.",
        "Z=.140/.049960=2.802243; two-sided p=.005075.",
        "Newcombe–Wilson 95% CI for d=[.042080,.234195]; RR=1.341, OR=1.759, h=.281154."
      ]),
      result: bi("A 组成功比例高 14.0 个百分点；区间表明相容差值约为 4.2–23.4 个百分点。", "Group A's success proportion is 14.0 percentage points higher; compatible differences are about 4.2–23.4 points."),
      approximate: true
    },
    inference: bi("检验的分母使用 H₀ 合并比例；估计风险差的区间不应复用该 H₀ 标准误。2×2 Pearson χ²（无连续性校正）与 Z² 等价。", "The test denominator uses the pooled null proportion; a risk-difference interval should not reuse that null SE. For a 2×2 table, uncorrected Pearson χ² equals Z²."),
    ci: bi("示例 Newcombe–Wilson 风险差 95% CI=[.0421,.2342]；它通常比简单 Wald CI [.0430,.2370] 更稳健。", "The example Newcombe–Wilson 95% CI for the risk difference is [.0421,.2342], generally more reliable than the simple Wald interval [.0430,.2370]."),
    effect: bi("风险差=.140，风险比=1.341，优势比=1.759，Cohen's h=.281；优先报告最符合研究问题的原始尺度效应及 CI。", "Risk difference=.140, risk ratio=1.341, odds ratio=1.759, and Cohen's h=.281; prioritize the raw-scale effect and CI that match the question."),
    edgeCases: bi([
      "配对或匹配比例必须使用 McNemar 或条件模型；忽略配对会得到错误 SE。",
      "零单元格、稀有事件或小期望数使用 Fisher、精确法或适当回归，不依赖 Wald。",
      "非劣效或等效检验的 H₀ 不是差值 0，需要预先给定界值和专门 score 方法，不能直接套此 pooled 公式。"
    ], [
      "Paired or matched proportions require McNemar or a conditional model; ignoring pairing gives the wrong SE.",
      "For zero cells, rare events, or small expected counts, use Fisher/exact inference or a suitable model rather than Wald inference.",
      "Noninferiority or equivalence has a nonzero null margin and needs a prespecified margin and dedicated score method; do not reuse this pooled-zero formula."
    ]),
    report: bi("A 组成功率为 55.0%（110/200），B 组为 41.0%（82/200）；两比例 Z=2.802，双侧 p=.0051，风险差=14.0 个百分点，Newcombe–Wilson 95% CI [4.2,23.4] 个百分点。", "Success was 55.0% (110/200) in A versus 41.0% (82/200) in B; two-proportion Z=2.802, two-sided p=.0051, risk difference=14.0 percentage points, Newcombe–Wilson 95% CI [4.2,23.4] points."),
    python: "from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep\ncount, nobs = [110,82], [200,200]\nz, p = proportions_ztest(count, nobs, value=0, alternative='two-sided', prop_var=False)\nci = confint_proportions_2indep(110,200,82,200, method='newcomb', compare='diff', alpha=0.05)\nprint(z, p, ci)",
    references: [OFFICIAL_REFS.proportionsZ, OFFICIAL_REFS.proportionsCI, OFFICIAL_REFS.nist],
    related: ["one_proportion_z", "chi_square_independence", "fisher_exact", "mcnemar", "logistic_regression"]
  });

  add({
    id: "chi_square_independence", family: "categorical", category: bi("分类计数", "Categorical counts"), outcome: "categorical", design: "independent", groups: "many", kind: "association",
    aliases: ["chi square independence", "chi2_contingency", "列联表卡方", "proportion test"], name: bi("卡方独立性检验", "Chi-square test of independence"),
    short: bi("比较列联表的观察计数与变量独立时的期望计数。", "Compares observed contingency-table counts with counts expected under independence."),
    useWhen: bi("两个分类变量来自独立对象，单元格保存人数/次数，且期望计数足以支持 χ² 近似。", "Use when two categorical variables are measured on independent units, cells contain counts, and expected counts support the χ² approximation."),
    h0: bi("两个分类变量独立；各组类别比例相同。", "The categorical variables are independent; category proportions are equal across groups."), h1: bi("两个变量有关；至少一个比例不同。", "The variables are associated; at least one proportion differs."),
    assumptions: bi(["每个独立对象只贡献一个单元格计数。", "输入原始计数而非百分比。", "常用经验是所有期望数≥1且至少 80%≥5；稀疏 2×2 表优先 Fisher。"], ["Each independent unit contributes to exactly one cell.", "Use raw counts, not percentages.", "A common rule is all expected counts ≥1 and at least 80% ≥5; use Fisher for sparse 2×2 tables."]),
    formulas: [
      { label: bi("期望计数", "Expected count"), tex: "E_{ij}=\\frac{(\\text{row }i\\text{ total})(\\text{column }j\\text{ total})}{N}" },
      { label: bi("Pearson χ²", "Pearson chi-square"), tex: "\\chi^2=\\sum_i\\sum_j\\frac{(O_{ij}-E_{ij})^2}{E_{ij}},\\quad df=(r-1)(c-1)" },
      { label: bi("Cramér's V", "Cramér's V"), tex: "V=\\sqrt{\\frac{\\chi^2}{N\\min(r-1,c-1)}}" }
    ], symbols: [{ symbol: "O_{ij},E_{ij}", meaning: bi("观察与期望计数", "observed and expected counts") }, { symbol: "r,c", meaning: bi("行、列类别数", "numbers of row and column categories") }],
    example: { caption: bi("班级与是否通过", "Class and pass status"), columns: bi(["组", "通过", "未通过", "合计"], ["Group", "Pass", "Fail", "Total"]), rows: [["A",18,7,25],["B",10,15,25],["合计 / Total",28,22,50]],
      steps: bi(["独立时每行期望为 [14,11]。", "χ²=Σ(O−E)²/E=5.195，df=1（未做 Yates 校正）。", "p≈.0227。", "Cramér's V=√(5.195/50)=.322。"], ["Under independence, each row's expected counts are [14,11].", "χ²=Σ(O−E)²/E=5.195, df=1 (without Yates correction).", "p≈.0227.", "Cramér's V=√(5.195/50)=.322."]), result: bi("班级与通过状态存在统计关联；应同时比较比例差与区间。", "Class and pass status are statistically associated; also report the proportion difference and interval."), approximate: true },
    inference: bi("χ² 近似检验整体独立性；2×2 表中不同软件可能默认 Yates 连续性校正，应明确设置。", "The χ² approximation tests overall independence. For 2×2 tables, software may default to Yates correction; state the choice."),
    ci: bi("2×2 表补充风险差、风险比或优势比及 CI；更大表可报告标准化残差。", "For 2×2 tables add a risk difference, risk ratio, or odds ratio with CI; for larger tables inspect adjusted residuals."),
    effect: bi("示例 Cramér's V=.322；它描述关联强度，不提供因果方向。", "Example Cramér's V=.322; it measures association strength, not causal direction."),
    report: bi("班级与通过状态有关，Pearson χ²(1,N=50)=5.195，p=.023，Cramér's V=.322（未做 Yates 校正）。", "Class was associated with pass status, Pearson χ²(1,N=50)=5.195, p=.023, Cramér's V=.322 (no Yates correction)."),
    python: "import numpy as np\nfrom scipy import stats\ntable = np.array([[18,7],[10,15]])\nchi2, p, dof, expected = stats.chi2_contingency(table, correction=False)\nprint(chi2, p, dof, expected)",
    related: ["two_proportion_z", "fisher_exact", "chi_square_goodness", "mcnemar"]
  });

  add({
    id: "chi_square_goodness", family: "categorical", category: bi("分类计数", "Categorical counts"), outcome: "categorical", design: "one", groups: "one", kind: "comparison",
    aliases: ["chi square goodness of fit", "chisquare", "拟合优度卡方"], name: bi("卡方拟合优度检验", "Chi-square goodness-of-fit test"),
    short: bi("检验一组类别计数是否符合预先指定的比例。", "Tests whether one set of category counts follows prespecified proportions."),
    useWhen: bi("每个独立对象属于一个互斥类别，预期比例在看数据前给定，期望计数足够大。", "Use when each independent unit belongs to one mutually exclusive category, expected proportions are prespecified, and expected counts are adequate."),
    h0: bi("总体类别概率等于指定向量 p₁,…,p_k。", "Population category probabilities equal the specified vector p₁,…,p_k."), h1: bi("至少一个类别概率不同。", "At least one category probability differs."),
    assumptions: bi(["类别互斥且穷尽，每个对象只计一次。", "期望概率不是用同一数据事后选择；若估计了 m 个参数，自由度减少 m。", "期望计数通常至少约 5；稀疏类别可按科学依据预先合并或用精确/模拟法。"], ["Categories are mutually exclusive and exhaustive; each unit is counted once.", "Expected probabilities are not chosen after seeing these data; subtract m fitted parameters from df.", "Expected counts are typically at least about 5; sparse categories may be prespecifically combined or tested exactly/by simulation."]),
    formulas: [
      { label: bi("期望与统计量", "Expected counts and statistic"), tex: "E_i=Np_i,\\quad \\chi^2=\\sum_{i=1}^k\\frac{(O_i-E_i)^2}{E_i}" },
      { label: bi("自由度", "Degrees of freedom"), tex: "df=k-1-m" },
      { label: bi("Cohen's w", "Cohen's w"), tex: "w=\\sqrt{\\sum_i\\frac{(\\hat p_i-p_i)^2}{p_i}}=\\sqrt{\\chi^2/N}" }
    ], symbols: [{ symbol: "p_i", meaning: bi("原假设中的类别概率", "null category probability") }, { symbol: "m", meaning: bi("由数据估计的参数数", "number of parameters estimated from the data") }],
    example: { caption: bi("四种颜色是否均匀", "Whether four colors are equally likely"), columns: bi(["类别", "观察 O", "期望 E"], ["Category", "Observed O", "Expected E"]), rows: [["红 / Red",18,15],["蓝 / Blue",12,15],["绿 / Green",20,15],["黄 / Yellow",10,15]],
      steps: bi(["N=60；均匀原假设给每类 E=15。", "χ²=(3²+(-3)²+5²+(-5)²)/15=4.533。", "df=4−1=3；p≈.209。", "Cohen's w=√(4.533/60)=.275。"], ["N=60; the uniform null gives E=15 per category.", "χ²=(3²+(-3)²+5²+(-5)²)/15=4.533.", "df=4−1=3; p≈.209.", "Cohen's w=√(4.533/60)=.275."]), result: bi("数据未提供足够证据否定均匀分布；这不证明完全均匀。", "The data do not provide enough evidence against uniformity; this does not prove exact uniformity."), approximate: true },
    inference: bi("用 χ²_df 近似；若预期概率由样本拟合，必须调整 df 或使用专门的拟合优度程序。", "Use the χ²_df approximation; if expected probabilities are fitted from the sample, adjust df or use a specialized goodness-of-fit procedure."),
    ci: bi("为各类别比例报告多项分布区间，并在多类别解释时控制同时覆盖率。", "Report multinomial proportion intervals, controlling simultaneous coverage when interpreting many categories."),
    effect: bi("示例 Cohen's w=.275；标准化残差显示哪些类别贡献最大。", "Example Cohen's w=.275; standardized residuals identify contributing categories."),
    report: bi("观察分布未显著偏离均匀分布，χ²(3,N=60)=4.533，p=.209，w=.275。", "Observed counts did not significantly depart from uniformity, χ²(3,N=60)=4.533, p=.209, w=.275."),
    python: "from scipy import stats\nobserved = [18,12,20,10]\nexpected = [15,15,15,15]\nprint(stats.chisquare(observed, f_exp=expected))",
    related: ["one_proportion_z", "chi_square_independence", "ks_test"]
  });

  add({
    id: "fisher_exact", family: "categorical", category: bi("分类计数", "Categorical counts"), outcome: "categorical", design: "independent", groups: "two", kind: "association",
    aliases: ["fisher exact", "fisher_exact", "2x2 exact", "费舍尔精确检验"], name: bi("Fisher 精确检验", "Fisher's exact test"),
    short: bi("在固定边际下精确检验稀疏 2×2 表的关联。", "Exactly tests association in a sparse 2×2 table conditional on fixed margins."),
    useWhen: bi("两个独立二分类变量构成小样本或低期望计数的 2×2 表。", "Use for a 2×2 table of two binary variables with independent units and small or sparse expected counts."),
    h0: bi("条件优势比 θ=1；在固定边际下无关联。", "The conditional odds ratio θ=1; no association given fixed margins."), h1: bi("θ≠1，或预先指定 θ>1 / θ<1。", "θ≠1, or a prespecified θ>1 / θ<1."),
    assumptions: bi(["对象独立，单元格是原始计数。", "经典检验条件于行列边际；研究设计应支持该条件解释。", "双侧“同样或更极端”的定义可能因软件而异。"], ["Units are independent and cells are raw counts.", "The classic test conditions on row and column margins; the design should support that interpretation.", "Definitions of equally or more extreme for a two-sided test can differ by software."]),
    formulas: [
      { label: bi("条件超几何概率", "Conditional hypergeometric probability"), tex: "P(A=a\\mid margins)=\\frac{\\binom{a+b}{a}\\binom{c+d}{c}}{\\binom{n}{a+c}}" },
      { label: bi("样本优势比", "Sample odds ratio"), tex: "\\widehat{OR}=\\frac{ad}{bc}" }
    ], symbols: [{ symbol: "a,b,c,d", meaning: bi("2×2 表四个单元格计数", "the four 2×2 cell counts") }, { symbol: "OR", meaning: bi("优势比", "odds ratio") }],
    example: { caption: bi("小样本治疗表", "Small treatment table"), columns: bi(["组", "有效", "无效"], ["Group", "Response", "No response"]), rows: [["治疗 / Treatment",8,1],["对照 / Control",1,8]],
      steps: bi(["边际固定为每行 9、每列 9，N=18。", "观察表的超几何概率为 81/48620。", "双侧概率排序还包括对称及更极端表，p=164/48620=.00337。", "样本 OR=(8×8)/(1×1)=64。"], ["Margins are fixed at 9 per row and column, N=18.", "The observed table's hypergeometric probability is 81/48620.", "Two-sided probability ordering includes the symmetric and more extreme tables, p=164/48620=.00337.", "Sample OR=(8×8)/(1×1)=64."]), result: bi("治疗反应与组别有关；OR 极大但小样本区间仍很宽。", "Treatment response is associated with group; the OR is huge but its small-sample interval remains wide."), approximate: false },
    inference: bi("精确 p 值来自所有具有相同边际的可能表。它不等于“参数无近似误差”的所有推断。", "The exact p-value enumerates tables with the same margins; it does not make every parameter estimate approximation-free."),
    ci: bi("报告条件精确 OR 区间；简单 Woolf 近似在本例约 [3.38,1212]，仅作说明。", "Report a conditional exact OR interval; the simple Woolf approximation is about [3.38,1212] here and is illustrative only."),
    effect: bi("OR=64；同时给两组风险 8/9 与 1/9、风险差和区间，便于实际解释。", "OR=64; also report risks 8/9 versus 1/9, their risk difference, and an interval for practical interpretation."),
    report: bi("Fisher 双侧精确检验显示反应与组别有关，p=.0034，OR=64；报告条件精确 95% CI。", "A two-sided Fisher exact test found an association, p=.0034, OR=64; a conditional exact 95% CI was reported."),
    python: "from scipy import stats\ntable = [[8,1],[1,8]]\nprint(stats.fisher_exact(table, alternative='two-sided'))",
    related: ["two_proportion_z", "chi_square_independence", "mcnemar"]
  });

  add({
    id: "mcnemar", family: "categorical", category: bi("配对分类", "Paired categorical"), outcome: "categorical", design: "paired", groups: "two", kind: "comparison",
    aliases: ["mcnemar test", "paired proportions", "配对四格表"], name: bi("McNemar 检验", "McNemar test"),
    short: bi("只用不一致配对检验同一对象前后二分类边际比例是否改变。", "Uses discordant pairs to test whether paired binary marginal proportions changed."),
    useWhen: bi("同一对象前后各有一个二分类结果，或二分类病例与对照一一匹配。", "Use when each unit has two paired binary outcomes, or binary cases and controls are matched one-to-one."),
    h0: bi("两个改变方向概率相等，即 P(1→0)=P(0→1)。", "The two change-direction probabilities are equal: P(1→0)=P(0→1)."), h1: bi("两个方向概率不同，或预先指定方向。", "The direction probabilities differ, or a direction is prespecified."),
    assumptions: bi(["每个对象只贡献一个完整配对；不同对象独立。", "重点是两个不一致格 b、c；一致格不提供边际变化信息。", "b+c 小时用精确二项版本；大样本可用 χ² 近似。"], ["Each unit contributes one complete pair; units are independent.", "Only discordant cells b and c inform marginal change.", "Use the exact binomial version when b+c is small; χ² approximation is suitable when large."]),
    formulas: [
      { label: bi("无连续性校正 χ²", "Uncorrected chi-square"), tex: "\\chi_M^2=\\frac{(b-c)^2}{b+c},\\quad df=1" },
      { label: bi("精确版本", "Exact version"), tex: "B\\mid(B+C=b+c)\\sim Binomial(b+c,0.5)" },
      { label: bi("配对优势比", "Matched odds ratio"), tex: "OR_{matched}=b/c" }
    ], symbols: [{ symbol: "b", meaning: bi("前阳后阴（按表方向定义）", "positive-to-negative discordant count, by table convention") }, { symbol: "c", meaning: bi("前阴后阳", "negative-to-positive discordant count") }],
    example: { caption: bi("前后阳性状态", "Positive status before and after"), columns: bi(["", "后阳", "后阴"], ["", "After +", "After −"]), rows: [["前阳 / Before +",12,8],["前阴 / Before −",2,18]],
      steps: bi(["不一致配对 b=8、c=2，b+c=10。", "无校正 χ²=(8−2)²/10=3.60，渐近 p≈.0578。", "因不一致对很少，优先精确二项：双侧 p=2P[Bin(10,.5)≤2]=.1094。", "匹配 OR=b/c=4。"], ["Discordant counts are b=8 and c=2, total 10.", "Uncorrected χ²=(8−2)²/10=3.60, asymptotic p≈.0578.", "Because discordant pairs are few, prefer the exact binomial result: two-sided p=.1094.", "Matched OR=b/c=4."]), result: bi("精确检验未达到 .05；旧式未校正渐近结果更小，但同样未显著。不能报告 p<.05。", "The exact test does not reach .05; the older uncorrected asymptotic result is smaller but also nonsignificant. It must not be reported as p<.05."), approximate: false },
    inference: bi("小样本主要报告精确 p；若用连续性校正或 mid-p，必须明确算法。", "For small samples report the exact p-value primarily; state any continuity correction or mid-p convention."),
    ci: bi("为配对 OR 或边际比例差报告配对设计专用区间；b 或 c 为 0 时常规 Wald 区间失效。", "Report a paired-design CI for the matched OR or marginal proportion difference; ordinary Wald intervals fail when b or c is zero."),
    effect: bi("示例 matched OR=4，但区间会很宽；同时报告 8 个改善与 2 个恶化。", "Example matched OR=4 with a wide interval; also report 8 improvements and 2 deteriorations."),
    report: bi("10 个不一致配对中 8 个阳转阴、2 个阴转阳；McNemar 双侧精确 p=.109，matched OR=4.0。", "Among 10 discordant pairs, 8 changed positive-to-negative and 2 the reverse; two-sided exact McNemar p=.109, matched OR=4.0."),
    python: "from statsmodels.stats.contingency_tables import mcnemar\ntable = [[12,8],[2,18]]\nprint(mcnemar(table, exact=True))",
    references: [OFFICIAL_REFS.statsmodels, OFFICIAL_REFS.nist], related: ["exact_sign_test", "fisher_exact", "chi_square_independence"]
  });

  add({
    id: "pearson_corr", family: "association", category: bi("相关与模型", "Association and models"), outcome: "continuous", design: "independent", groups: "two", kind: "association",
    aliases: ["pearson r", "pearsonr", "product moment correlation", "皮尔逊相关"], name: bi("Pearson 相关", "Pearson correlation"),
    short: bi("量化两个连续变量的线性关系。", "Quantifies the linear association between two continuous variables."),
    useWhen: bi("每个独立对象有一对连续观测，散点图关系近似线性，且无主导结果的高杠杆离群点。", "Use when each independent unit supplies two continuous values, the scatter is approximately linear, and no high-leverage outlier dominates."),
    h0: bi("总体线性相关 ρ=0。", "Population linear correlation ρ=0."), h1: bi("ρ≠0，或预先指定正/负方向。", "ρ≠0, or a prespecified positive/negative direction."),
    assumptions: bi(["观测对之间独立，X 与 Y 成对完整。", "关系近似线性；r 不描述一般非线性关系。", "经典小样本 t 推断通常假设二元正态；务必查看散点图和影响点。"], ["Pairs are independent and X/Y are jointly observed.", "The relationship is approximately linear; r does not capture arbitrary nonlinear association.", "Classical small-sample t inference assumes bivariate normality; inspect the scatterplot and influential points."]),
    formulas: [
      { label: bi("样本相关", "Sample correlation"), tex: "r=\\frac{\\sum_i(x_i-\\bar x)(y_i-\\bar y)}{\\sqrt{\\sum_i(x_i-\\bar x)^2\\sum_i(y_i-\\bar y)^2}}" },
      { label: bi("检验统计量", "Test statistic"), tex: "t=r\\sqrt{\\frac{n-2}{1-r^2}},\\quad df=n-2" },
      { label: bi("Fisher z 区间", "Fisher z interval"), tex: "z=\\operatorname{atanh}(r),\\quad SE_z=1/\\sqrt{n-3}" }
    ], symbols: [{ symbol: "r,\\rho", meaning: bi("样本、总体 Pearson 相关", "sample and population Pearson correlations") }, { symbol: "n", meaning: bi("完整观测对数", "number of complete pairs") }],
    example: { caption: bi("学习时长与成绩", "Study time and score"), columns: bi(["对象", "小时 X", "成绩 Y"], ["Unit", "Hours X", "Score Y"]), rows: [[1,1,60],[2,2,65],[3,3,70],[4,4,78],[5,5,82]],
      steps: bi(["x̄=3，ȳ=71；Σ(x−x̄)(y−ȳ)=57。", "Σ(x−x̄)²=10，Σ(y−ȳ)²=328。", "r=57/√3280=.9953。", "t≈17.73，df=3，双侧 p≈.0004；Fisher 95% CI≈[.927,.9997]。"], ["x̄=3 and ȳ=71; Σ(x−x̄)(y−ȳ)=57.", "Σ(x−x̄)²=10 and Σ(y−ȳ)²=328.", "r=57/√3280=.9953.", "t≈17.73, df=3, two-sided p≈.0004; Fisher 95% CI≈[.927,.9997]."]), result: bi("样本呈很强正线性关系；相关不等于因果。", "The sample shows a very strong positive linear association; correlation is not causation."), approximate: true },
    inference: bi("t 检验 H₀:ρ=0；若关注任意依赖或非线性关系，应选其他模型。", "The t test targets H₀:ρ=0; use another model for arbitrary dependence or nonlinear structure."),
    ci: bi("对 r 做 Fisher z 变换构造 CI，再用 tanh 变回相关尺度。", "Construct the CI on Fisher's z scale and transform back with tanh."),
    effect: bi("r 本身是带方向效应量；r²=.991 表示样本线性共享变异比例，但不是因果解释度。", "r is a directional effect size; r²=.991 is the sample linear shared-variance fraction, not causal variance explained."),
    report: bi("学习时长与成绩呈正线性相关，r(3)=.995，p<.001，95% CI [.927,1.000]，n=5。", "Study time and score were positively correlated, r(3)=.995, p<.001, 95% CI [.927,1.000], n=5."),
    python: "from scipy import stats\nx = [1,2,3,4,5]\ny = [60,65,70,78,82]\nresult = stats.pearsonr(x, y)\nprint(result, result.confidence_interval())",
    related: ["spearman_corr", "linear_regression"]
  });

  add({
    id: "spearman_corr", family: "association", category: bi("相关与模型", "Association and models"), outcome: "ordinal", design: "independent", groups: "two", kind: "association",
    aliases: ["spearman rho", "spearmanr", "rank correlation", "斯皮尔曼相关"], name: bi("Spearman 相关", "Spearman rank correlation"),
    short: bi("用秩量化两个变量的单调关系。", "Quantifies monotonic association using ranks."),
    useWhen: bi("变量至少有序，或连续关系单调但不线性；希望降低极端数值幅度的影响。", "Use when variables are at least ordinal or continuously related monotonically but not linearly, and magnitude outliers should have less influence."),
    h0: bi("总体秩相关 ρ_s=0（在相应置换原假设下无关联）。", "Population rank correlation ρ_s=0 (no association under the corresponding permutation null)."), h1: bi("ρ_s≠0，或预先指定方向。", "ρ_s≠0, or a prespecified direction."),
    assumptions: bi(["观测对之间独立。", "变量至少可排序；并列使用平均秩。", "关系应为单调；U 形关系可能得到接近 0 的 ρ_s。"], ["Pairs are independent.", "Variables are rankable; ties receive average ranks.", "The relationship should be monotonic; a U-shaped relation can yield ρ_s near zero."]),
    formulas: [
      { label: bi("一般定义", "General definition"), tex: "\\rho_s=Corr(rank(X),rank(Y))" },
      { label: bi("无并列简式", "No-tie shortcut"), tex: "r_s=1-\\frac{6\\sum_i d_i^2}{n(n^2-1)}" }
    ], symbols: [{ symbol: "d_i", meaning: bi("第 i 对的两个秩之差", "difference between the two ranks for pair i") }, { symbol: "r_s", meaning: bi("样本 Spearman 相关", "sample Spearman correlation") }],
    example: { caption: bi("满意度与复购意愿等级", "Satisfaction and repurchase-intent ranks"), columns: bi(["对象", "X", "Y"], ["Unit", "X", "Y"]), rows: [[1,1,1],[2,2,2],[3,3,3],[4,4,5],[5,5,4]],
      steps: bi(["秩已经是 1–5；差 d 为 0,0,0,−1,1。", "Σd²=2。", "r_s=1−6×2/[5(25−1)]=.90。", "常用 t 近似给双侧 p≈.037；n=5 时优先精确置换 p，并说明算法。"], ["Values are already ranks 1–5; d values are 0,0,0,−1,1.", "Σd²=2.", "r_s=1−6×2/[5(25−1)]=.90.", "The common t approximation gives two-sided p≈.037; with n=5 prefer an exact permutation p-value and state the algorithm."]), result: bi("单调关联很强，但极小样本推断对精确/近似算法敏感。", "The monotonic association is strong, but tiny-sample inference is sensitive to exact versus approximate algorithms."), approximate: true },
    inference: bi("大样本可用渐近近似；小样本用置换枚举，并列时使用能处理并列的算法。", "Use asymptotic inference for large samples; for small samples enumerate permutations with appropriate tie handling."),
    ci: bi("通过对象级 bootstrap 或适当变换获得区间；极小样本 bootstrap 也可能不稳定。", "Obtain an interval by unit-level bootstrap or an appropriate transformation; bootstrap intervals can also be unstable in tiny samples."),
    effect: bi("ρ_s=.90 是带方向效应量，描述单调强度而非每增加 1 单位的变化量。", "ρ_s=.90 is a directional effect size for monotonic strength, not a per-unit change."),
    report: bi("两等级变量呈正单调关联，Spearman r_s=.90；小样本置换 p 值及 95% bootstrap CI 另行报告。", "The ordinal variables had a positive monotonic association, Spearman r_s=.90; the small-sample permutation p-value and 95% bootstrap CI were reported."),
    python: "from scipy import stats\nx = [1,2,3,4,5]\ny = [1,2,3,5,4]\nprint(stats.spearmanr(x, y))  # asymptotic p; use permutation_test for tiny n",
    related: ["pearson_corr", "mann_whitney"]
  });

  add({
    id: "linear_regression", family: "model", category: bi("相关与模型", "Association and models"), outcome: "continuous", design: "independent", groups: "many", kind: "association",
    aliases: ["ordinary least squares", "ols", "线性模型"], name: bi("线性回归", "Linear regression"),
    short: bi("用一个或多个自变量解释连续结果的条件均值。", "Models the conditional mean of a continuous outcome using one or more predictors."),
    useWhen: bi("结果连续，关注协变量调整后的平均变化或预测；均值结构可合理写成线性组合。", "Use for a continuous outcome when adjusted mean changes or predictions are sought and the mean structure is plausibly linear."),
    h0: bi("某个回归系数 β_j=0（控制模型中其他变量）。", "A coefficient β_j=0 conditional on the other model variables."), h1: bi("β_j≠0，或预先指定方向。", "β_j≠0, or a prespecified direction."),
    assumptions: bi(["对象独立；聚类/重复数据需相关误差模型。", "E(Y|X) 的函数形式正确，误差方差用于经典 SE 时近似恒定。", "残差近似正态只影响小样本 t/F 推断；检查异常、高杠杆和有影响点。"], ["Units are independent; clustered/repeated data need a correlated-error model.", "The functional form of E(Y|X) is correct; classical SEs assume roughly constant error variance.", "Residual normality matters for small-sample t/F inference; inspect outliers, leverage, and influence."]),
    formulas: [
      { label: bi("模型", "Model"), tex: "Y_i=\\beta_0+\\beta_1X_{i1}+\\cdots+\\beta_pX_{ip}+\\varepsilon_i" },
      { label: bi("最小二乘", "Least squares"), tex: "\\hat\\beta=(X^TX)^{-1}X^Ty,\\quad t_j=\\hat\\beta_j/SE(\\hat\\beta_j)" },
      { label: bi("决定系数", "Coefficient of determination"), tex: "R^2=1-\\frac{SS_{res}}{SS_{tot}}" }
    ], symbols: [{ symbol: "\\beta_j", meaning: bi("控制其他预测变量后的条件均值系数", "conditional mean coefficient adjusting for other predictors") }, { symbol: "\\varepsilon_i", meaning: bi("未解释误差", "unexplained error") }],
    example: { caption: bi("运动小时预测血压", "Exercise hours predicting blood pressure"), columns: bi(["对象", "X 小时", "Y mmHg"], ["Unit", "X hours", "Y mmHg"]), rows: [[1,1,142],[2,2,139],[3,3,133],[4,4,131],[5,5,126]],
      steps: bi(["x̄=3，ȳ=134.2，Sxx=10，Sxy=−40。", "斜率 b₁=Sxy/Sxx=−4.00；截距 b₀=146.2。", "SSE=2.8，SE(b₁)=√[(2.8/3)/10]=.306。", "t=−13.09，df=3，p≈.001；斜率 95% CI≈[−4.97,−3.03]；R²=.983。"], ["x̄=3, ȳ=134.2, Sxx=10, and Sxy=−40.", "Slope b₁=Sxy/Sxx=−4.00; intercept b₀=146.2.", "SSE=2.8 and SE(b₁)=√[(2.8/3)/10]=.306.", "t=−13.09, df=3, p≈.001; slope 95% CI≈[−4.97,−3.03]; R²=.983."]), result: bi("样本中每多 1 小时，条件平均血压降低约 4 mmHg；观察研究不能自动作因果解释。", "In this sample, each additional hour corresponds to about 4 mmHg lower conditional mean pressure; observational data do not automatically support causality."), approximate: true },
    inference: bi("系数 t 检验是模型条件下的局部推断；模型整体可用 F 检验。异方差时使用稳健 SE。", "Coefficient t tests are model-conditional; use an F test for the overall model. Use robust SEs under heteroskedasticity."),
    ci: bi("报告系数 CI；区分均值响应 CI 与新个体预测区间，后者更宽。", "Report coefficient CIs and distinguish confidence intervals for the mean from wider prediction intervals for new units."),
    effect: bi("报告原始斜率、标准化系数（若有意义）、partial R² 或模型 R²。", "Report raw slopes, standardized coefficients when meaningful, partial R², or model R²."),
    report: bi("运动小时负向预测血压，b=−4.00 mmHg/小时，SE=.31，t(3)=−13.09，p≈.001，95% CI [−4.97,−3.03]，R²=.983。", "Exercise hours negatively predicted pressure, b=−4.00 mmHg/hour, SE=.31, t(3)=−13.09, p≈.001, 95% CI [−4.97,−3.03], R²=.983."),
    python: "import statsmodels.api as sm\nx = sm.add_constant([1,2,3,4,5])\ny = [142,139,133,131,126]\nfit = sm.OLS(y, x).fit()\nprint(fit.summary())\nprint(fit.conf_int())",
    references: [OFFICIAL_REFS.statsmodels, OFFICIAL_REFS.nist], related: ["pearson_corr", "logistic_regression", "poisson_regression"]
  });

  add({
    id: "logistic_regression", family: "model", category: bi("相关与模型", "Association and models"), outcome: "categorical", design: "independent", groups: "many", kind: "association",
    aliases: ["binary logistic", "logit", "glm binomial", "逻辑回归"], name: bi("Logistic 回归", "Logistic regression"),
    short: bi("用线性预测器建模二分类事件的对数优势。", "Models the log-odds of a binary event with a linear predictor."),
    useWhen: bi("独立对象的结果为 0/1，需要估计预测变量与事件优势或概率的调整后关系。", "Use for independent binary outcomes when adjusted associations with event odds or probabilities are needed."),
    h0: bi("某个 β_j=0，等价于调整后 OR_j=exp(β_j)=1。", "A coefficient β_j=0, equivalently adjusted OR_j=exp(β_j)=1."), h1: bi("β_j≠0，等价于 OR_j≠1。", "β_j≠0, equivalently OR_j≠1."),
    assumptions: bi(["对象独立，事件编码和参考类别清楚。", "连续预测变量与 logit 的关系形式正确；可用样条处理非线性。", "每个参数有足够事件；完全/准分离时普通 MLE 发散，应使用惩罚或偏差修正。"], ["Units are independent, with clear event coding and reference categories.", "Continuous predictors have a correct functional relationship with the logit; splines can model nonlinearity.", "There are enough events per parameter; ordinary MLE diverges under complete/quasi separation, requiring penalization or bias correction."]),
    formulas: [
      { label: bi("Logit 模型", "Logit model"), tex: "\\log\\frac{p_i}{1-p_i}=\\beta_0+\\sum_j\\beta_jX_{ij}" },
      { label: bi("优势比", "Odds ratio"), tex: "OR_j=e^{\\beta_j},\\quad CI_{OR}=e^{\\hat\\beta_j\\pm z_{1-\\alpha/2}SE(\\hat\\beta_j)}" },
      { label: bi("Wald 统计量", "Wald statistic"), tex: "z_j=\\hat\\beta_j/SE(\\hat\\beta_j)" }
    ], symbols: [{ symbol: "p_i", meaning: bi("第 i 个对象的事件概率", "event probability for unit i") }, { symbol: "OR_j", meaning: bi("X_j 增加一单位的条件优势比", "conditional odds ratio for a one-unit increase in X_j") }],
    example: { caption: bi("培训与事件（避免完全分离）", "Training and event outcome (without complete separation)"), columns: bi(["培训", "事件", "无事件", "总数"], ["Training", "Event", "No event", "Total"]), rows: [["否 / No",3,7,10],["是 / Yes",1,9,10]],
      steps: bi(["未培训事件优势=3/7；培训事件优势=1/9。", "β̂=log[(1/9)/(3/7)]=−1.350；OR=.259。", "SE(β̂)=√(1/1+1/9+1/3+1/7)=1.260。", "Wald z=−1.07，p≈.284；OR 95% CI≈[.022,3.07]。"], ["Untrained event odds=3/7; trained event odds=1/9.", "β̂=log[(1/9)/(3/7)]=−1.350; OR=.259.", "SE(β̂)=√(1/1+1/9+1/3+1/7)=1.260.", "Wald z=−1.07, p≈.284; OR 95% CI≈[.022,3.07]."]), result: bi("点估计提示培训组优势较低，但样本太小、区间很宽。这个示例没有完全分离。", "The estimate suggests lower odds with training, but the sample is tiny and the interval wide. This example avoids complete separation."), approximate: true },
    inference: bi("可用 Wald、似然比或 score 检验；稀疏数据优先似然比、Firth 或惩罚估计而非依赖 Wald。", "Wald, likelihood-ratio, or score tests are possible; for sparse data prefer likelihood or Firth/penalized methods over a fragile Wald test."),
    ci: bi("OR 区间由 β 区间指数变换。概率效应依赖基线风险，建议补充预测概率或边际风险差。", "Exponentiate the β interval for an OR interval. Probability effects depend on baseline risk, so add predicted probabilities or marginal risk differences."),
    effect: bi("示例 OR=.259；OR 不是风险比，事件常见时两者差别可很大。", "Example OR=.259; an odds ratio is not a risk ratio and can differ substantially when outcomes are common."),
    edgeCases: bi(["完全分离会产生巨大系数/SE 或不收敛；不要把它当作极强的普通显著结果。", "小样本或稀有事件考虑 Firth、Bayesian 或正则化模型。", "评估校准、区分度和外部验证，不能只报告 AUC。"], ["Complete separation causes huge coefficients/SEs or nonconvergence; do not treat it as an ordinary extremely significant effect.", "For small or rare-event data consider Firth, Bayesian, or regularized models.", "Assess calibration, discrimination, and external validation rather than AUC alone."]),
    report: bi("培训的事件 OR=.259，Wald z=−1.07，p=.284，95% CI [.022,3.07]；样本稀疏，结论不确定。", "The event OR for training was .259, Wald z=−1.07, p=.284, 95% CI [.022,3.07]; sparse data make the estimate uncertain."),
    python: "import statsmodels.api as sm\ntraining = [0]*10 + [1]*10\ny = [1,1,1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0,0]\nX = sm.add_constant(training)\nfit = sm.Logit(y, X).fit()\nprint(fit.summary())\nprint('OR and CI:', __import__('numpy').exp(fit.params), __import__('numpy').exp(fit.conf_int()))",
    references: [OFFICIAL_REFS.statsmodels, OFFICIAL_REFS.nist], related: ["chi_square_independence", "linear_regression", "poisson_regression"]
  });

  add({
    id: "poisson_regression", family: "model", category: bi("相关与模型", "Association and models"), outcome: "count", design: "independent", groups: "many", kind: "association",
    aliases: ["poisson glm", "count regression", "rate regression", "泊松广义线性模型"], name: bi("泊松回归", "Poisson regression"),
    short: bi("建模计数或带暴露量的事件发生率。", "Models counts or event rates with an exposure term."),
    useWhen: bi("结果为非负计数，观察窗口明确；不同暴露量通过 log(offset) 加入。", "Use for nonnegative counts with defined observation windows; include log exposure as an offset when windows differ."),
    h0: bi("某个 β_j=0，等价于发生率比 IRR_j=1。", "A coefficient β_j=0, equivalently incidence-rate ratio IRR_j=1."), h1: bi("β_j≠0，等价于 IRR_j≠1。", "β_j≠0, equivalently IRR_j≠1."),
    assumptions: bi(["给定协变量后计数独立，暴露量正确且为正。", "条件均值满足 log(E[Y|X]) 的线性结构。", "标准 Poisson 假设条件方差等于均值；过度离散时用稳健 SE、准 Poisson 或负二项。"], ["Counts are conditionally independent and exposure is positive and correctly measured.", "The log conditional mean has the specified linear form.", "Standard Poisson assumes conditional variance equals the mean; use robust SEs, quasi-Poisson, or negative binomial under overdispersion."]),
    formulas: [
      { label: bi("率模型", "Rate model"), tex: "\\log E(Y_i)=\\log(exposure_i)+\\beta_0+\\sum_j\\beta_jX_{ij}" },
      { label: bi("发生率比", "Incidence-rate ratio"), tex: "IRR_j=e^{\\beta_j}" },
      { label: bi("两组简单 SE", "Simple two-group SE"), tex: "SE(\\log IRR)\\approx\\sqrt{1/Y_1+1/Y_0}" }
    ], symbols: [{ symbol: "exposure_i", meaning: bi("观察时间或风险暴露量", "observation time or amount at risk") }, { symbol: "IRR", meaning: bi("发生率比", "incidence-rate ratio") }],
    example: { caption: bi("每人观察 6 个月的事故次数", "Accidents over six months per worker"), columns: bi(["培训", "个体计数", "总事故", "总人月"], ["Training", "Individual counts", "Total events", "Person-months"]), rows: [["否 / No","4,3,5,2",14,24],["是 / Yes","1,2,0,1",4,24]],
      steps: bi(["未培训率=14/24=.583/月；培训率=4/24=.167/月。", "IRR=.167/.583=.286；β̂=log(.286)=−1.253。", "SE≈√(1/14+1/4)=.567。", "z=−2.21，p≈.027；IRR 95% CI≈[.094,.868]。"], ["Untrained rate=14/24=.583/month; trained rate=4/24=.167/month.", "IRR=.167/.583=.286; β̂=log(.286)=−1.253.", "SE≈√(1/14+1/4)=.567.", "z=−2.21, p≈.027; IRR 95% CI≈[.094,.868]."]), result: bi("培训组观察发生率较低；还需检查过度离散和混杂。", "The trained group has a lower observed rate; assess overdispersion and confounding."), approximate: true },
    inference: bi("系数可用 Wald 或似然比检验；用 Pearson χ²/df、deviance 和残差检查离散度与拟合。", "Use Wald or likelihood-ratio tests and inspect Pearson χ²/df, deviance, and residuals for dispersion and fit."),
    ci: bi("对 β 区间指数变换得到 IRR 区间；同时报告每组率及其区间。", "Exponentiate the β interval for an IRR interval and also report group-specific rates with intervals."),
    effect: bi("示例 IRR=.286，表示培训组估计发生率约为未培训组的 28.6%。", "Example IRR=.286, meaning the estimated trained-group rate is 28.6% of the untrained rate."),
    edgeCases: bi(["方差远大于均值提示过度离散；普通 Poisson SE 可能过小。", "零过多可能来自异质性或结构机制，先理解数据生成过程再选零膨胀模型。", "重复事件或聚类对象需要 GEE、随机效应或适当稳健 SE。"], ["Variance much larger than the mean indicates overdispersion and overly small ordinary Poisson SEs.", "Excess zeros can arise from heterogeneity or structure; understand the data process before selecting a zero-inflated model.", "Repeated events or clusters require GEE, random effects, or suitable robust SEs."]),
    report: bi("培训与较低事故率相关，IRR=.286，z=−2.21，p=.027，95% CI [.094,.868]；模型包含 log(人月) offset。", "Training was associated with a lower accident rate, IRR=.286, z=−2.21, p=.027, 95% CI [.094,.868], with log person-months as an offset."),
    python: "import numpy as np\nimport statsmodels.api as sm\ny = np.array([4,3,5,2,1,2,0,1])\ntraining = np.array([0,0,0,0,1,1,1,1])\nexposure = np.repeat(6, 8)\nfit = sm.GLM(y, sm.add_constant(training), family=sm.families.Poisson(), offset=np.log(exposure)).fit()\nprint(fit.summary())\nprint('IRR:', np.exp(fit.params), np.exp(fit.conf_int()))",
    references: [OFFICIAL_REFS.statsmodels, OFFICIAL_REFS.nist], related: ["logistic_regression", "linear_regression"]
  });

  add({
    id: "ks_test", family: "distribution", category: bi("分布检验", "Distribution tests"), outcome: "continuous", design: "independent", groups: "two", kind: "comparison",
    aliases: ["kolmogorov smirnov", "ks_2samp", "kstest", "ks检验"], name: bi("Kolmogorov–Smirnov 检验", "Kolmogorov–Smirnov test"),
    short: bi("比较一条经验累计分布与理论分布，或两条经验累计分布的最大距离。", "Compares the maximum distance between an empirical CDF and a theoretical CDF, or between two empirical CDFs."),
    useWhen: bi("连续分布的整体差异是目标；一元理论分布参数在检验前已知，或比较两个独立连续样本。", "Use when overall continuous-distribution differences are the target and one-sample reference parameters are known in advance, or for two independent continuous samples."),
    h0: bi("一元：F=F₀；两样本：F₁=F₂。", "One-sample: F=F₀; two-sample: F₁=F₂."), h1: bi("累计分布在至少一点不同，或预先指定随机序方向。", "The CDFs differ somewhere, or a stochastic-order direction is prespecified."),
    assumptions: bi(["样本内和样本间观测独立。", "经典 KS 假设连续分布；大量并列/离散数据需置换或专门检验。", "若 F₀ 的参数由同一数据估计，普通 KS 临界值无效（如 Lilliefors 问题）。"], ["Observations are independent within and between samples.", "Classical KS assumes continuous distributions; many ties or discrete data need permutation or specialized tests.", "If F₀ parameters are estimated from the same data, ordinary KS critical values are invalid (the Lilliefors problem)."]),
    formulas: [
      { label: bi("一元统计量", "One-sample statistic"), tex: "D_n=\\sup_x|F_n(x)-F_0(x)|" },
      { label: bi("两样本统计量", "Two-sample statistic"), tex: "D_{n,m}=\\sup_x|F_n(x)-G_m(x)|" }
    ], symbols: [{ symbol: "F_n,G_m", meaning: bi("经验累计分布函数", "empirical cumulative distribution functions") }, { symbol: "\\sup_x", meaning: bi("所有 x 上的最大值", "maximum over all x") }],
    example: { caption: bi("两个独立小样本", "Two independent small samples"), columns: bi(["A", "B"], ["A", "B"]), rows: [[1.1,1.8],[1.3,2.0],[1.5,2.2],[1.8,2.4],[2.0,2.7]],
      steps: bi(["分别构造每次跳跃 0.2 的经验 CDF。", "在 x=1.5 后 F_A=.6、F_B=0；在 x=2.0 处差距也达到 .6。", "最大绝对差 D=.60。", "n=m=5 的双侧精确 p≈.357。"], ["Construct ECDFs with jumps of 0.2.", "Just after x=1.5, F_A=.6 and F_B=0; the gap also reaches .6 at x=2.0.", "The maximum absolute gap is D=.60.", "For n=m=5, the two-sided exact p≈.357."]), result: bi("未发现整体分布差异；小样本只有很粗的 D 分辨率。", "No overall distribution difference is detected; tiny samples give coarse D resolution."), approximate: true },
    inference: bi("软件根据样本量选择精确或渐近分布；报告 method。KS 对分布中部较敏感，对尾部差异未必最有力。", "Software chooses exact or asymptotic inference by sample size; report the method. KS is not especially powerful for tail-only differences."),
    ci: bi("KS 通常报告 D 和 p；可用 DKW 界构造整条 CDF 的同时置信带。", "KS usually reports D and p; DKW bounds can provide simultaneous confidence bands for the full CDF."),
    effect: bi("D=.60 是两条经验 CDF 的最大概率差，本身可作尺度无关差异描述。", "D=.60 is the maximum probability-scale gap between ECDFs and is a scale-free discrepancy measure."),
    report: bi("两样本 KS 检验未发现分布差异，D=.60，双侧精确 p=.357，n₁=n₂=5。", "A two-sample KS test did not detect a distribution difference, D=.60, two-sided exact p=.357, n₁=n₂=5."),
    python: "from scipy import stats\na = [1.1,1.3,1.5,1.8,2.0]\nb = [1.8,2.0,2.2,2.4,2.7]\nprint(stats.ks_2samp(a, b, alternative='two-sided', method='exact'))",
    related: ["shapiro_wilk", "mann_whitney", "chi_square_goodness"]
  });

  add({
    id: "shapiro_wilk", family: "diagnostic", category: bi("前提检查", "Assumption checks"), outcome: "continuous", design: "one", groups: "one", kind: "diagnostic",
    aliases: ["shapiro", "normality test", "正态性检验"], name: bi("Shapiro–Wilk 正态性检验", "Shapiro–Wilk normality test"),
    short: bi("用有序样本与正态次序统计量的匹配程度检查正态性。", "Assesses normality through the match between ordered data and expected normal order statistics."),
    useWhen: bi("把它作为 Q–Q 图和领域判断的辅助诊断；配对 t 检查差值，回归检查残差。", "Use as a supplement to Q–Q plots and domain judgment; assess paired differences for paired t and residuals for regression."),
    h0: bi("样本来自某个正态分布。", "The sample comes from a normal distribution."), h1: bi("样本分布不是正态。", "The sample distribution is not normal."),
    assumptions: bi(["输入为独立连续观测或恰当模型残差。", "对检验目标使用正确对象：差值/残差，而非机械检查每列。", "大样本会检出无关紧要的微小偏离；小样本又可能功效不足。"], ["Input consists of independent continuous observations or appropriate model residuals.", "Assess the correct object—differences or residuals—not every raw column mechanically.", "Large samples detect trivial deviations, while small samples may lack power."]),
    formulas: [
      { label: bi("W 统计量", "W statistic"), tex: "W=\\frac{(\\sum_{i=1}^n a_i x_{(i)})^2}{\\sum_{i=1}^n(x_i-\\bar x)^2}" }
    ], symbols: [{ symbol: "x_{(i)}", meaning: bi("第 i 个有序观测值", "ith ordered observation") }, { symbol: "a_i", meaning: bi("由正态次序统计量均值与协方差决定的系数", "coefficients derived from normal order-statistic means and covariance") }],
    example: { caption: bi("6 个近似对称成绩", "Six roughly symmetric scores"), columns: bi(["观测"], ["Observation"]), rows: [[78],[82],[85],[79],[86],[81]],
      steps: bi(["排序为 78,79,81,82,85,86。", "软件用 n=6 对应的 Shapiro–Wilk 系数组合有序值。", "W 约 .94；精确数值依赖标准算法实现。", "p>.05，未发现明显偏离正态，但不能证明正态。"], ["Order values as 78,79,81,82,85,86.", "Software combines ordered values using Shapiro–Wilk coefficients for n=6.", "W is about .94; the precise value follows the standard algorithm implementation.", "p>.05, so no clear departure is detected, but normality is not proven."]), result: bi("结合 Q–Q 图看起来没有严重偏离；不应据此机械决定 t 与非参数方法。", "Together with a Q–Q plot there is no severe departure; do not mechanically choose t versus nonparametric methods from this p-value."), approximate: true },
    inference: bi("小 p 表示数据与正态模型不相容；大 p 只表示未检出偏离。研究设计、估计目标、离群点和样本量更重要。", "A small p indicates incompatibility with normality; a large p only means no departure was detected. Design, estimand, outliers, and sample size matter more."),
    ci: bi("正态性检验没有“正态程度”的常规 CI；使用 Q–Q 图置信带或对目标估计做稳健敏感性分析。", "There is no routine CI for degree of normality; use Q–Q bands or robust sensitivity analyses for the target estimate."),
    effect: bi("W 是诊断统计量，不宜套用固定“小/中/大”效应阈值。", "W is a diagnostic statistic, not an effect size with fixed small/medium/large cutoffs."),
    edgeCases: bi(["重复值、离散量尺和截断数据会改变零分布。", "对大数据，图形和对实际推断稳健性的评估比“是否显著”更有用。", "正态性不等于方差齐性或独立性。"], ["Ties, discrete scales, and truncation alter the null behavior.", "For large data, plots and robustness of the target inference are more useful than significance alone.", "Normality does not imply equal variance or independence."]),
    report: bi("Shapiro–Wilk 检验未检出明显非正态，W≈.94，p>.05；同时检查 Q–Q 图且未见严重离群点。", "Shapiro–Wilk did not detect clear nonnormality, W≈.94, p>.05; the Q–Q plot showed no severe outliers."),
    python: "from scipy import stats\nx = [78,82,85,79,86,81]\nprint(stats.shapiro(x))  # pairwise work: pass differences; regression: inspect residuals",
    related: ["one_sample_t", "paired_t", "levene_test", "ks_test"]
  });

  add({
    id: "levene_test", family: "diagnostic", category: bi("前提检查", "Assumption checks"), outcome: "continuous", design: "independent", groups: "many", kind: "diagnostic",
    aliases: ["levene", "brown forsythe", "variance homogeneity", "方差齐性检验"], name: bi("Levene / Brown–Forsythe 检验", "Levene / Brown–Forsythe test"),
    short: bi("把各观测到组中心的绝对偏差做 ANOVA，检查组间方差。", "Applies ANOVA to absolute deviations from group centers to assess equal variance."),
    useWhen: bi("独立组连续结果需要描述或诊断方差齐性；中位数中心的 Brown–Forsythe 版本对非正态更稳健。", "Use to diagnose equal variance across independent continuous groups; the median-centered Brown–Forsythe version is more robust to nonnormality."),
    h0: bi("所有总体方差相等。", "All population variances are equal."), h1: bi("至少一个总体方差不同。", "At least one population variance differs."),
    assumptions: bi(["组间和组内对象独立。", "结果至少为区间尺度；组中心可选均值、中位数或截尾均值。", "检验不显著不能证明方差相等，且不应机械决定 pooled 与 Welch。"], ["Units are independent across and within groups.", "Outcomes are at least interval-scale; centers may be means, medians, or trimmed means.", "Nonsignificance does not prove equal variances and should not mechanically select pooled over Welch methods."]),
    formulas: [
      { label: bi("绝对偏差", "Absolute deviations"), tex: "Z_{ij}=|Y_{ij}-\\tilde Y_j|" },
      { label: bi("对 Z 做单因素 ANOVA", "One-way ANOVA on Z"), tex: "F=\\frac{(N-k)\\sum_j n_j(\\bar Z_j-\\bar Z)^2}{(k-1)\\sum_j\\sum_i(Z_{ij}-\\bar Z_j)^2}" }
    ], symbols: [{ symbol: "\\tilde Y_j", meaning: bi("第 j 组中位数（Brown–Forsythe）", "group-j median for Brown–Forsythe") }, { symbol: "Z_{ij}", meaning: bi("到组中心的绝对偏差", "absolute deviation from the group center") }],
    example: { caption: bi("三组不同散布", "Three groups with different spreads"), columns: bi(["组", "数值", "中位数"], ["Group", "Values", "Median"]), rows: [["A","8,9,10,11",9.5],["B","7,12,16,20",14],["C","9,10,11,12",10.5]],
      steps: bi(["绝对偏差均值分别为 1、4.25、1。", "对 12 个绝对偏差做 ANOVA：SS_B=28.167，SS_W=22.75。", "F=(28.167/2)/(22.75/9)=5.571。", "df=(2,9)，p≈.027。"], ["Mean absolute deviations are 1, 4.25, and 1.", "ANOVA on 12 deviations gives SS_B=28.167 and SS_W=22.75.", "F=(28.167/2)/(22.75/9)=5.571.", "df=(2,9), p≈.027."]), result: bi("组 B 的散布更大；后续均值比较优先 Welch 方法，并直接报告各组 SD/稳健散布。", "Group B is more variable; prefer Welch methods for mean comparisons and directly report group SDs or robust spreads."), approximate: true },
    inference: bi("p 检验方差齐性原假设，但选择 Welch 不需要先让 Levene 显著；设计和方差/样本量不平衡同样重要。", "The p-value tests equal variance, but Welch does not require a significant Levene test; design and imbalance also matter."),
    ci: bi("检验本身不给方差比 CI；可分别报告 SD/方差区间或 bootstrap 方差比区间。", "The test itself gives no variance-ratio CI; report group SD/variance intervals or bootstrap variance-ratio intervals."),
    effect: bi("报告绝对偏差均值差或方差比，不把 Levene F 当作实际方差差的完整效应量。", "Report differences in mean absolute deviation or variance ratios rather than treating Levene F as a complete effect size."),
    report: bi("中位数中心 Brown–Forsythe 检验显示方差不同，F(2,9)=5.571，p=.027；后续使用 Welch 方法。", "A median-centered Brown–Forsythe test indicated unequal variances, F(2,9)=5.571, p=.027; Welch methods were used subsequently."),
    python: "from scipy import stats\na = [8,9,10,11]\nb = [7,12,16,20]\nc = [9,10,11,12]\nprint(stats.levene(a, b, c, center='median'))",
    related: ["welch_t", "welch_anova", "shapiro_wilk"]
  });

  const CONCEPTS = {
    cohens_dz: {
      id: "cohens_dz", kind: "effect", family: "effect", category: bi("效应量", "Effect sizes"),
      aliases: ["s_dz", "dz", "cohens dz", "cohens_dz", "paired standardized effect size"],
      formulas: [
        { label: bi("配对差值", "Paired differences"), tex: "D_i=Y_{i,after}-Y_{i,before},\\quad \\bar D=\\frac{1}{n}\\sum_iD_i" },
        { label: bi("Cohen's d_z", "Cohen's d_z"), tex: "d_z=\\frac{\\bar D}{s_D}=\\frac{t}{\\sqrt n}" }
      ],
      symbols: [
        { symbol: "\\bar D", meaning: bi("配对差值均值", "mean paired difference") },
        { symbol: "s_D", meaning: bi("配对差值的样本标准差", "sample SD of paired differences") },
        { symbol: "n", meaning: bi("完整配对数", "number of complete pairs") }
      ],
      example: { caption: bi("四个前后差值", "Four before–after differences"), columns: bi(["对象", "D"], ["Pair", "D"]), rows: [[1,-2],[2,-4],[3,-3],[4,-3]], steps: bi(["D̄=(−2−4−3−3)/4=−3。", "s_D=0.816。", "d_z=−3/0.816≈−3.67。", "若来自配对 t，t=d_z√n≈−7.35。"], ["D̄=(−2−4−3−3)/4=−3.", "s_D=0.816.", "d_z=−3/0.816≈−3.67.", "If paired t is used, t=d_z√n≈−7.35."]), result: bi("d_z 保留差值方向；它量化变化大小，不提供独立 p 值。", "d_z keeps the direction of the difference and quantifies its size; it does not produce an independent p-value.") },
      zh: { name: "Cohen's d_z（配对效应量）", short: "配对差值均值除以配对差值标准差。", background: "配对 t 检验把每一对压缩成一个差值；d_z 将平均变化用差值波动标准化。", useWhen: "报告配对 t 或重复测量的标准化变化时。", avoidWhen: "不要把 d_z 与独立组 pooled d、d_av 或 d_rm 混用；必须写清分母。", hypotheses: { h0: "效应量本身不定义 H₀；通常与配对差值检验一起报告。", h1: "同上。" }, assumptions: ["差值方向必须预先定义。", "效应量不替代研究设计、区间和原始单位。"], formulaNotes: "d_z 的绝对值越大表示变化相对差值波动越大；正负号取决于差值方向。", inference: "p 值来自配对 t/Wilcoxon/符号检验，而不是 d_z 本身。", ci: "可用非中心 t 方法或 bootstrap 为 d_z 构造区间。", effect: "d_z 本身就是效应量；同时报告原始平均差和其置信区间。", edgeCases: ["n 很小且差值高度一致时，d_z 可能极大。", "跨研究比较前确认效应量定义一致。"], report: "配对差值均值为 −3.0，配对效应量 Cohen's d_z=−3.67。", references: [OFFICIAL_REFS.scipy, OFFICIAL_REFS.nist], related: ["paired_t", "wilcoxon_signed"] },
      en: { name: "Cohen's d_z (paired effect size)", short: "The mean paired difference divided by its sample SD.", background: "A paired t-test reduces each pair to one difference; d_z standardizes the mean change by the variability of those differences.", useWhen: "Report a standardized change alongside a paired t-test or repeated-measures comparison.", avoidWhen: "Do not mix d_z with independent-group pooled d, d_av, or d_rm; state the denominator.", hypotheses: { h0: "An effect size does not define H₀; it is reported alongside a paired-difference test.", h1: "Same.\"" }, assumptions: ["Prespecify the direction of the difference.", "An effect size does not replace design, intervals, or raw-unit interpretation."], formulaNotes: "The absolute value reflects standardized magnitude; the sign follows the difference direction.", inference: "The p-value comes from the paired t, Wilcoxon, or sign test, not from d_z itself.", ci: "Use a noncentral-t or bootstrap method for a d_z interval.", effect: "d_z is the effect size; also report the raw mean difference and CI.", edgeCases: ["With tiny n and nearly identical differences, d_z can be very large.", "Confirm the effect-size definition before comparing studies."], report: "The mean paired difference was −3.0, with Cohen's d_z=−3.67.", references: [OFFICIAL_REFS.scipy, OFFICIAL_REFS.nist], related: ["paired_t", "wilcoxon_signed"] }
    }
  };

  const DECISION_TREE = {
    start: { title: bi("你要比较、关联，还是检查模型前提？", "What is your goal: compare, associate, or check assumptions?"), options: [
      { label: bi("比较一组或多组结果", "Compare one or more groups"), desc: bi("均值、秩次或分类比例的差异。", "Compare means, ranks, or categorical proportions."), next: "compare" },
      { label: bi("研究两个变量的关系或预测", "Study a relationship or prediction"), desc: bi("相关、连续预测、二分类或计数结果。", "Correlation, continuous prediction, binary or count outcomes."), next: "association" },
      { label: bi("检查分布或分析前提", "Check a distribution or assumption"), desc: bi("正态性、方差齐性或拟合程度。", "Normality, equal variance, or goodness of fit."), next: "diagnostic" }
    ] },
    compare: { title: bi("结果变量属于哪一类？", "What kind of outcome do you have?"), options: [
      { label: bi("连续数值", "Continuous numeric"), desc: bi("成绩、血压、时间、收入等。", "Scores, blood pressure, time, income, etc."), next: "continuous" },
      { label: bi("有序等级", "Ordinal rating"), desc: bi("满意度、疼痛等级或分期。", "Satisfaction, pain ratings, or stages."), next: "ordinal" },
      { label: bi("分类/计数", "Categorical/count"), desc: bi("是否、类别、次数或事件数。", "Yes/no, categories, counts, or events."), next: "categorical" }
    ] },
    continuous: { title: bi("连续结果的设计是什么？", "What is the continuous-outcome design?"), options: [
      { label: bi("一组均值 vs 预先给定值", "One mean vs a prespecified value"), desc: bi("先判断总体标准差 σ 是否由本样本之外可靠地已知；不能用样本标准差 s 冒充 σ。", "First determine whether population SD σ is reliably known independently of this sample; sample SD s is not a known σ."), next: "one_mean_conditions" },
      { label: bi("两组独立", "Two independent groups"), desc: bi("根据方差与分布情况选择 Student、Welch 或秩检验。", "Choose Student, Welch, or a rank test from the variance and distribution evidence."), next: "two_continuous" },
      { label: bi("配对/前后测", "Paired or before–after"), desc: bi("同一对象或一一匹配的两次测量。", "The same unit or a genuine matched pair."), next: "paired_shape" },
      { label: bi("三组以上独立", "Three or more independent groups"), desc: bi("根据方差与分布情况选择 ANOVA、Welch ANOVA 或 Kruskal–Wallis。", "Choose ANOVA, Welch ANOVA, or Kruskal–Wallis from the assumptions."), next: "many_continuous" },
      { label: bi("三次以上配对/重复测量", "Three or more paired/repeated measurements"), desc: bi("比较重复测量均值或秩。", "Compare repeated-measure means or ranks."), next: "repeated_continuous" }
    ] },
    one_mean_conditions: { title: bi("总体标准差 σ 是否在查看本样本前已由外部资料可靠给定？", "Was population SD σ reliably specified externally before this sample was examined?"), options: [
      { label: bi("σ 已知，且均值的正态模型或近似合理", "σ is known, with a suitable normal model or approximation for the mean"), desc: bi("观测还应独立；使用单样本均值 Z 检验，不能用本样本的 s 代替 σ。", "Observations must also be independent; use the one-sample mean Z-test and do not substitute this sample's s for σ."), result: "one_sample_z" },
      { label: bi("σ 未知，需要用本样本估计", "σ is unknown and must be estimated from this sample"), desc: bi("使用单样本 t 检验；大样本只会让 t 分布接近 Z，并不会使 σ 变成已知。", "Use the one-sample t-test; a large sample makes t approach Z but does not make σ known."), result: "one_sample_t" }
    ] },
    two_continuous: { title: bi("两组独立连续结果满足什么条件？", "Which conditions fit two independent continuous groups?"), options: [
      { label: bi("方差近似相等且均值模型合理", "Comparable variances and a suitable mean model"), desc: bi("使用合并方差的独立样本 t 检验。", "Use the pooled-variance independent t-test."), result: "independent_t" },
      { label: bi("方差不等或不确定，但关心均值", "Variances unequal or uncertain; means matter"), desc: bi("通常优先使用 Welch t 检验。", "Usually prefer Welch's t-test."), result: "welch_t" },
      { label: bi("有序、明显偏态或只信任秩", "Ordinal, strongly skewed, or rank-focused"), desc: bi("使用 Mann–Whitney U；它不是均值检验。", "Use Mann–Whitney U; it is not a test of means."), result: "mann_whitney" }
    ] },
    many_continuous: { title: bi("三组以上独立结果满足什么条件？", "Which conditions fit three or more independent groups?"), options: [
      { label: bi("方差近似相等且均值模型合理", "Comparable variances and a suitable mean model"), desc: bi("使用经典单因素 ANOVA。", "Use classical one-way ANOVA."), result: "one_way_anova" },
      { label: bi("方差不等或样本量不平衡", "Unequal variances or unbalanced sizes"), desc: bi("使用 Welch ANOVA。", "Use Welch ANOVA."), result: "welch_anova" },
      { label: bi("有序、明显偏态或只信任秩", "Ordinal, strongly skewed, or rank-focused"), desc: bi("使用 Kruskal–Wallis。", "Use Kruskal–Wallis."), result: "kruskal_wallis" }
    ] },
    repeated_continuous: { title: bi("三次以上重复测量如何建模？", "How should three or more repeated measurements be modeled?"), options: [
      { label: bi("连续结果，关心条件均值", "Continuous outcome; condition means matter"), desc: bi("使用重复测量 ANOVA，并检查球形性。", "Use repeated-measures ANOVA and assess sphericity."), result: "repeated_anova" },
      { label: bi("有序/非参数，关心秩差异", "Ordinal/nonparametric; ranks matter"), desc: bi("使用 Friedman 检验。", "Use the Friedman test."), result: "friedman" }
    ] },
    paired_shape: { title: bi("配对差值更接近哪种情况？", "Which description fits the paired differences?"), options: [
      { label: bi("差值近似正态，关心平均变化", "Approximately normal; mean change matters"), desc: bi("使用配对 t 检验。", "Use the paired t-test."), result: "paired_t" },
      { label: bi("大致对称，差值大小的秩有意义", "Roughly symmetric; ranks of magnitude matter"), desc: bi("使用 Wilcoxon 符号秩检验。", "Use the Wilcoxon signed-rank test."), result: "wilcoxon_signed" },
      { label: bi("明显不对称/离群，只相信方向", "Strongly asymmetric/outlying; direction only"), desc: bi("使用精确配对符号检验。", "Use the exact paired sign test."), result: "exact_sign_test" }
    ] },
    ordinal: { title: bi("有序结果如何配对？", "How are the ordinal outcomes related?"), options: [
      { label: bi("两组独立", "Two independent groups"), desc: bi("使用 Mann–Whitney U。", "Use Mann–Whitney U."), result: "mann_whitney" },
      { label: bi("两次配对", "Two paired measurements"), desc: bi("使用 Wilcoxon 或精确符号检验。", "Use Wilcoxon or the exact sign test."), next: "paired_shape" },
      { label: bi("三组以上独立", "Three or more independent groups"), desc: bi("使用 Kruskal–Wallis。", "Use Kruskal–Wallis."), result: "kruskal_wallis" },
      { label: bi("三次以上重复测量", "Three or more repeated measures"), desc: bi("使用 Friedman 检验。", "Use the Friedman test."), result: "friedman" }
    ] },
    categorical: { title: bi("分类/计数结果的设计或目标是什么？", "What is the categorical/count design or target?"), options: [
      { label: bi("一组二分类比例 vs 预设比例 p₀", "One binary proportion vs a prespecified p₀"), desc: bi("例如通过率、患病率或发生率与研究前给定基准比较。", "Compare a pass, prevalence, or event rate with a benchmark specified before analysis."), next: "one_proportion_conditions" },
      { label: bi("两组独立的二分类比例", "Two independent binary proportions"), desc: bi("比较两个不重叠组的成功率；配对或前后数据不属于此分支。", "Compare rates in two nonoverlapping groups; paired or before–after data do not belong here."), next: "two_proportion_conditions" },
      { label: bi("同一对象前后或匹配的二分类结果", "Paired or matched binary outcomes"), desc: bi("使用 McNemar 检验，不要当作两个独立比例。", "Use McNemar's test rather than treating the outcomes as independent proportions."), result: "mcnemar" },
      { label: bi("两个分类变量的整体关联（r×c）", "Overall association between two categorical variables (r×c)"), desc: bi("期望频数合适时使用 Pearson 卡方独立性检验。", "Use Pearson's chi-square test of independence when expected counts are adequate."), result: "chi_square_independence" },
      { label: bi("一组多类别频数 vs 预设比例向量", "One multicategory distribution vs a prespecified proportion vector"), desc: bi("使用卡方拟合优度检验。", "Use the chi-square goodness-of-fit test."), result: "chi_square_goodness" },
      { label: bi("预测二分类结果", "Predict a binary outcome"), desc: bi("使用 Logistic 回归。", "Use logistic regression."), result: "logistic_regression" },
      { label: bi("预测计数结果", "Predict a count outcome"), desc: bi("使用泊松回归。", "Use Poisson regression."), result: "poisson_regression" }
    ] },
    one_proportion_conditions: { title: bi("H₀ 下成功与失败的期望计数是否足以支持正态近似？", "Are the expected success and failure counts under H₀ adequate for a normal approximation?"), options: [
      { label: bi("np₀ 与 n(1−p₀) 都不稀疏", "Both np₀ and n(1−p₀) are non-sparse"), desc: bi("检查 H₀ 下两类期望计数，而不只看总样本量；数值阈值只是经验线。", "Check both null expected counts, not sample size alone; numerical cutoffs are only rules of thumb."), result: "one_proportion_z" },
      { label: bi("任一期望计数很小，或 p₀ 接近 0/1", "Either expected count is small, or p₀ is near 0 or 1"), desc: bi("不要依赖 Z 近似。", "Do not rely on the Z approximation."), note: bi("使用精确二项检验，并报告适合小样本的比例区间；Python 可用 scipy.stats.binomtest。", "Use an exact binomial test with a small-sample-appropriate interval; in Python, use scipy.stats.binomtest.") }
    ] },
    two_proportion_conditions: { title: bi("两组是否独立，且 H₀:p₁=p₂ 下四个期望计数都不稀疏？", "Are the groups independent and all four expected counts under H₀:p₁=p₂ non-sparse?"), options: [
      { label: bi("独立，且四个 H₀ 期望计数都不稀疏", "Independent groups with four non-sparse null expected counts"), desc: bi("用合并比例检查两组成功与失败期望数；近似合理时使用两比例 Z。", "Use the pooled proportion to check expected successes and failures in both groups; use the two-proportion Z-test when the approximation is defensible."), result: "two_proportion_z" },
      { label: bi("2×2 表稀疏、出现零格，或近似可疑", "The 2×2 table is sparse, has a zero cell, or the approximation is doubtful"), desc: bi("不要依赖两比例 Z；使用 Fisher 精确检验。", "Do not rely on the two-proportion Z approximation; use Fisher's exact test."), result: "fisher_exact" },
      { label: bi("其实是配对、匹配或同一对象前后数据", "The observations are actually paired, matched, or before–after"), desc: bi("改用 McNemar 检验。", "Use McNemar's test instead."), result: "mcnemar" }
    ] },
    association: { title: bi("你的关系/预测问题是哪一种？", "Which relationship or prediction question fits?"), options: [
      { label: bi("两个连续变量的线性关系", "Linear relation between two numeric variables"), desc: bi("使用 Pearson 相关。", "Use Pearson correlation."), result: "pearson_corr" },
      { label: bi("等级或单调关系", "Ordinal or monotonic relation"), desc: bi("使用 Spearman 相关。", "Use Spearman correlation."), result: "spearman_corr" },
      { label: bi("预测连续结果", "Predict a continuous outcome"), desc: bi("使用线性回归。", "Use linear regression."), result: "linear_regression" },
      { label: bi("预测二分类结果", "Predict a binary outcome"), desc: bi("使用 Logistic 回归。", "Use logistic regression."), result: "logistic_regression" },
      { label: bi("预测计数结果", "Predict a count outcome"), desc: bi("使用泊松回归。", "Use Poisson regression."), result: "poisson_regression" }
    ] },
    diagnostic: { title: bi("你要检查什么？", "What do you need to check?"), options: [
      { label: bi("单变量是否近似正态", "Whether one variable is approximately normal"), desc: bi("使用 Shapiro–Wilk，并结合 Q–Q 图。", "Use Shapiro–Wilk together with a Q–Q plot."), result: "shapiro_wilk" },
      { label: bi("两组/多组方差是否相近", "Whether group variances are comparable"), desc: bi("使用 Levene/Brown–Forsythe。", "Use Levene/Brown–Forsythe."), result: "levene_test" },
      { label: bi("连续样本是否来自指定分布", "Whether a continuous sample follows a specified distribution"), desc: bi("使用 KS 检验；参数应预先指定或做相应校正。", "Use KS; parameters must be prespecified or appropriately adjusted."), result: "ks_test" },
      { label: bi("多类别频数是否符合预设比例向量", "Whether multicategory counts match a prespecified proportion vector"), desc: bi("使用卡方拟合优度检验。", "Use the chi-square goodness-of-fit test."), result: "chi_square_goodness" }
    ] }
  };

  const CONCEPT_ENTRIES = Object.values(CONCEPTS);
  const HELP_CONTENT = [
    { id: "start", title: bi("如何使用这个工具", "How to use this guide"), body: bi(["先明确结果变量、研究单位、配对关系和目标参数，再使用向导。", "若已经知道方法，直接打开方法库或搜索别名。", "页面给出教学示例；正式分析仍需回到研究设计、缺失值和数据诊断。"], ["Define the outcome, unit of analysis, pairing, and estimand before using the guide.", "If you already know the method, open the library or search an alias.", "Examples are for learning; formal analysis still requires design, missing-data, and diagnostic review."]) },
    { id: "pvalue", title: bi("p 值、置信区间和效应量", "p-values, intervals, and effect sizes"), body: bi(["p 值是在 H₀ 下得到当前或更极端统计量的概率，不是 H₀ 为真的概率。", "置信区间描述估计的精度和与哪些效应相容；不应只看是否跨过 0。", "效应量说明差异大小，必须结合原始单位和实际重要性。"], ["A p-value is the probability of a statistic at least as extreme under H₀, not the probability that H₀ is true.", "A confidence interval conveys precision and compatible effects; do not read it only as a zero-crossing rule.", "Effect sizes describe magnitude and must be interpreted with raw units and practical importance."]) },
    { id: "design", title: bi("独立、配对与分析单位", "independent, paired, and unit of analysis"), body: bi(["同一对象前后测量或自然匹配的数据必须保留配对关系。", "配对方法要求不同配对之间独立，而不是要求同一对象的两次测量独立。", "重复使用同一对象、对话或数据集时，分析单位应与研究问题一致。"], ["Keep the pairing for before–after or genuinely matched data.", "Paired methods require independence between pairs, not independence of the two observations within a pair.", "When a unit is reused, the analysis unit must match the research question."]) },
    { id: "assumptions", title: bi("前提、ties 和缺失值", "assumptions, ties, and missing values"), body: bi(["不要把单个正态性检验当作自动路由器；看差值/残差图形、离群点、样本量和估计目标。", "Wilcoxon 的精确零分布受零差和并列秩影响；符号检验要报告删除的 ties。", "缺失值不能默认为零；提前写出 complete-case、插补或模型处理规则。"], ["Do not use one normality-test p-value as an automatic router; inspect differences/residuals, outliers, sample size, and estimand.", "Zeros and tied ranks change Wilcoxon exact null distributions; report ties removed by the sign test.", "Missing values are not zeros; prespecify complete-case, imputation, or model-based handling."]) },
    { id: "multiple", title: bi("多重比较和报告", "multiple comparisons and reporting"), body: bi(["同一研究反复检验会提高假阳性；考虑预先指定主要比较并控制 FWER 或 FDR。", "报告估计值、95% CI、统计量、自由度、p 值、样本量、效应量和方向。", "用“未拒绝 H₀”而不是“接受 H₀”；统计显著不等于实际重要。"], ["Repeated tests inflate false positives; prespecify primary comparisons and control FWER or FDR.", "Report the estimate, 95% CI, statistic, df, p-value, sample size, effect size, and direction.", "Say “failed to reject H₀,” not “accepted H₀”; statistical significance is not practical importance."]) },
    { id: "glossary", title: bi("常用术语", "Common terms"), body: bi(["连续变量：可以在区间内取小数的数值结果。", "有序变量：类别有自然顺序，但间距不一定相等。", "精确检验：直接使用离散零分布，不依赖大样本近似。", "效应量：描述差异/关联幅度的指标。"], ["Continuous: numeric outcomes that can take values on an interval.", "Ordinal: ordered categories whose spacing need not be equal.", "Exact test: uses a discrete null distribution rather than a large-sample approximation.", "Effect size: a measure of the magnitude of a difference or association."]) }
  ];

  const P_ALPHA_TEXT = {
    zh: {
      eyebrow: "互动专题 · p 值与 α",
      title: "先看零分布，再理解 p 值",
      lead: "p 值并不都来自正态分布。选择不同检验族，观察零分布、极端区域和判定如何一起变化。",
      distribution: "零分布示例",
      normal: "Z / 标准正态（Z 检验、大样本近似）",
      student: "t(5)（小样本 t 检验示意）",
      chiSquare: "χ²(4)（卡方检验示意）",
      binomial: "Binomial(12, 0.5)（精确符号检验）",
      alternative: "备择方向",
      twoSided: "双侧：任一方向都算极端",
      greater: "右侧：越大越极端",
      less: "左侧：越小越极端",
      observed: "观测统计量",
      successes: "正差对数 k",
      alpha: "显著性水平 α（分析前设定）",
      pValue: "当前 p 值",
      decision: "当前判定",
      reject: "p ≤ α：在这个 α 水平拒绝 H₀",
      retain: "p > α：证据不足，未拒绝 H₀",
      pMeaning: "蓝色区域是：假设 H₀ 和模型前提成立，零分布中与观测结果同样或更极端的概率。",
      alphaMeaning: "红色边界标出 p ≤ α 的拒绝域。α 不改变 p 值；它只改变你事先采用的判定门槛。",
      discreteMeaning: "离散检验由一根根概率柱组成，p 值是若干柱子的概率之和，不是连续曲线下面积。",
      relationTitle: "一句话分清三件事",
      nullDistribution: "零分布：如果 H₀ 成立，统计量会怎样波动；它由检验方法决定，不一定是正态。",
      pDefinition: "p 值：在这个零分布里，从观测统计量向“更极端”方向累计的概率。",
      alphaDefinition: "α：看数据前选定的长期假阳性风险上限；常用 0.05 只是惯例，不是自然定律。",
      interpretation: "正确读法",
      interpretationText: "例如 p = 0.03：若 H₀ 与模型前提成立，得到当前或更极端结果的概率为 3%。它不表示 H₀ 有 3% 概率为真。",
      simulateTitle: "在 H₀ 真的情况下重复 100 次",
      simulateLead: "每个圆点是一项独立研究。红色 × 表示该次 p ≤ α，属于假阳性；长期错误率由 α 控制，连续检验通常接近 α，离散精确检验可能更保守。",
      run: "运行 100 次",
      running: "正在抽样…",
      falsePositives: "假阳性",
      expected: "长期预期约",
      notAllNormal: "为什么不都用正态分布？",
      familyMap: [
        "均值类检验常见 t 或近似正态零分布。",
        "列联表、方差分析等会使用 χ² 或 F 零分布。",
        "符号检验、Fisher 精确检验等直接使用离散的精确零分布。",
        "Wilcoxon 等秩检验使用由秩和产生的零分布。"
      ],
      cautions: "四个常见误解",
      cautionItems: [
        "p < 0.05 不表示有 95% 概率结论是真的。",
        "p > 0.05 不证明 H₀ 为真，也不证明“没有差异”。",
        "更小的 p 值不等于更大的效应；还要看估计值、置信区间和效应量。",
        "先看 p 值再选择单侧方向或 α 会破坏原先的错误率含义。"
      ],
      chartLabel: "零分布与极端区域示意图",
      observedLabel: "观测值",
      pRegionLabel: "p 值区域",
      rejectionLabel: "α 拒绝域",
      simulationLabel: "100 次零假设重复抽样结果",
      noneYet: "点击“运行 100 次”查看 α 的长期含义。"
    },
    en: {
      eyebrow: "Interactive topic · p-values and α",
      title: "Start with the null distribution",
      lead: "P-values do not all come from a normal distribution. Switch test families to see how the null distribution, extreme region, and decision change together.",
      distribution: "Example null distribution",
      normal: "Z / standard normal (Z tests, large-sample approximations)",
      student: "t(5) (small-sample t-test illustration)",
      chiSquare: "χ²(4) (chi-square test illustration)",
      binomial: "Binomial(12, 0.5) (exact sign test)",
      alternative: "Alternative direction",
      twoSided: "Two-sided: either direction is extreme",
      greater: "Right-tailed: larger is more extreme",
      less: "Left-tailed: smaller is more extreme",
      observed: "Observed statistic",
      successes: "Number of positive differences k",
      alpha: "Significance level α (set before analysis)",
      pValue: "Current p-value",
      decision: "Current decision",
      reject: "p ≤ α: reject H₀ at this α level",
      retain: "p > α: insufficient evidence; fail to reject H₀",
      pMeaning: "The blue region is the probability, assuming H₀ and the model assumptions, of a statistic at least as extreme as observed.",
      alphaMeaning: "The red boundary marks the rejection region where p ≤ α. Alpha does not change the p-value; it only changes the prespecified decision threshold.",
      discreteMeaning: "A discrete exact test has probability bars. Its p-value is a sum of selected bar probabilities, not an area under a continuous curve.",
      relationTitle: "Separate three ideas in one minute",
      nullDistribution: "Null distribution: how the statistic varies if H₀ is true. The test determines it; it need not be normal.",
      pDefinition: "P-value: probability accumulated from the observed statistic toward outcomes defined as more extreme in that null distribution.",
      alphaDefinition: "Alpha: a prespecified long-run false-positive limit. The conventional 0.05 is a convention, not a law of nature.",
      interpretation: "Correct reading",
      interpretationText: "For example, p = 0.03 means that, if H₀ and the model assumptions hold, a result at least as extreme occurs with probability 3%. It does not mean H₀ has a 3% chance of being true.",
      simulateTitle: "Repeat 100 studies while H₀ is true",
      simulateLead: "Each dot is an independent study. A red × is a false positive with p ≤ α. Alpha controls the long-run rate; continuous tests are often near α, while discrete exact tests can be conservative.",
      run: "Run 100 studies",
      running: "Sampling…",
      falsePositives: "False positives",
      expected: "Long-run expectation ≈",
      notAllNormal: "Why not always a normal distribution?",
      familyMap: [
        "Mean-based tests often use t or approximately normal null distributions.",
        "Contingency tables and variance analyses may use χ² or F null distributions.",
        "Sign and Fisher exact tests use discrete exact null distributions.",
        "Rank tests such as Wilcoxon use null distributions generated by rank sums."
      ],
      cautions: "Four common misconceptions",
      cautionItems: [
        "p < 0.05 does not mean the conclusion has a 95% probability of being true.",
        "p > 0.05 neither proves H₀ nor proves there is no difference.",
        "A smaller p-value does not imply a larger effect; inspect the estimate, interval, and effect size.",
        "Choosing a one-sided direction or α after seeing p breaks the intended error-rate interpretation."
      ],
      chartLabel: "Illustration of a null distribution and extreme regions",
      observedLabel: "Observed",
      pRegionLabel: "p-value region",
      rejectionLabel: "α rejection region",
      simulationLabel: "Results from 100 repeated samples under the null",
      noneYet: "Choose “Run 100 studies” to see the long-run meaning of α."
    }
  };

  const P_ALPHA_STORY = {
    zh: {
      eyebrow: "从零开始 · 一个完整例子",
      title: "12 个人里有 10 个人变好，这说明什么？",
      lead: "假设一项训练前后各测一次。排除零差后有 12 对数据，其中 10 人提高、2 人下降。我们用精确符号检验，从研究问题一步走到 p 值。",
      stepLabel: "步骤",
      previous: "上一步",
      next: "下一步",
      restart: "重新开始",
      steps: [
        {
          title: "1. 先把问题说清楚",
          body: ["同一个人训练前后各有一个分数，因此数据是配对的。", "这里暂时不使用提高了多少分，只记录方向：提高记作 +，下降记作 −。", "研究问题是：训练后出现正向变化的概率是否高于 50%？"],
          callout: "先确定研究单位、配对关系和方向，再谈 p 值。"
        },
        {
          title: "2. 写出 H₀ 和 H₁",
          body: ["H₀ 是用来计算参照概率的假设：训练没有系统性方向，正差和负差各有 50% 机会。", "H₁ 是事先提出的研究方向：正差出现的概率大于 50%。"],
          formula: "H_0:\\pi_+=0.5,\\qquad H_1:\\pi_+>0.5",
          callout: "H₀ 不是我们相信它为真，而是先假定它成立，看看当前数据有多反常。"
        },
        {
          title: "3. 看数据前选 α",
          body: ["这里预先选择 α=0.05。它是一条决策规则，不是从这批数据算出来的。", "它表示：如果 H₀ 真的成立，并反复执行同样的完整研究流程，错误拒绝 H₀ 的长期概率控制在约 5% 或以下。"],
          formula: "\\alpha=0.05",
          callout: "α 不是“H₀ 有 5% 概率为真”，也不是“这个结论有 5% 概率出错”。"
        },
        {
          title: "4. 观察数据并计算 p 值",
          body: ["实际观察到 K=10 个正差。若 H₀ 成立，则 K 服从 Binomial(12,0.5)。", "因为 H₁ 事先规定为“正差更多”，所以更极端是 K=10、11、12。把这三种概率相加就是单侧 p 值。"],
          formula: "p=P_0(K\\ge10)=\\frac{{12\\choose10}+{12\\choose11}+{12\\choose12}}{2^{12}}=\\frac{79}{4096}=0.0193",
          callout: "p=0.0193 的意思是：在 H₀ 与抽样前提成立时，12 人中至少 10 人提高的概率约为 1.93%。"
        },
        {
          title: "5. 比较 p 和 α，再谨慎下结论",
          body: ["因为 0.0193 < 0.05，所以在预先选定的 5% 水平拒绝 H₀。", "可以说数据提供了正向变化超过 50% 的证据；不能说 H₁ 有 98.07% 概率为真。", "还应报告 10/12 的原始比例、置信区间、零差数量，并讨论变化是否具有实际意义。"],
          formula: "p=0.0193<\\alpha=0.05\\quad\\Longrightarrow\\quad \\text{拒绝 }H_0",
          callout: "一句话：p 描述这次结果在 H₀ 参照分布中的尾部位置；α 是看数据前画好的决策边界。"
        }
      ]
    },
    en: {
      eyebrow: "From the beginning · one complete example",
      title: "Ten of twelve people improved—what does that tell us?",
      lead: "Suppose scores are measured before and after training. After excluding zero differences, 10 of 12 pairs improved and 2 declined. We use the exact sign test to walk from the question to a p-value.",
      stepLabel: "Step",
      previous: "Previous",
      next: "Next",
      restart: "Start again",
      steps: [
        {
          title: "1. State the question",
          body: ["Each person has a before and after score, so the data are paired.", "For now we ignore the magnitude and record only direction: improvement is + and decline is −.", "The question is whether the probability of a positive change exceeds 50%."],
          callout: "Define the analysis unit, pairing, and direction before discussing a p-value."
        },
        {
          title: "2. Write H₀ and H₁",
          body: ["H₀ supplies the reference probabilities: training has no systematic direction, so positive and negative differences each have probability 50%.", "H₁ is the prespecified research direction: the probability of a positive difference is greater than 50%."],
          formula: "H_0:\\pi_+=0.5,\\qquad H_1:\\pi_+>0.5",
          callout: "We do not have to believe H₀; we assume it temporarily to ask how unusual the data would be."
        },
        {
          title: "3. Choose α before seeing the data",
          body: ["Here we prespecify α=0.05. It is a decision rule, not something calculated from this dataset.", "If H₀ is true and the complete study procedure is repeated, the long-run probability of wrongly rejecting H₀ is controlled at about 5% or below."],
          formula: "\\alpha=0.05",
          callout: "Alpha does not mean H₀ has a 5% chance of being true or that this particular conclusion has a 5% chance of being wrong."
        },
        {
          title: "4. Observe the data and calculate p",
          body: ["We observe K=10 positive differences. Under H₀, K follows Binomial(12,0.5).", "Because H₁ prespecified 'more positives,' outcomes K=10, 11, and 12 are at least as extreme. Their probabilities sum to the one-sided p-value."],
          formula: "p=P_0(K\\ge10)=\\frac{{12\\choose10}+{12\\choose11}+{12\\choose12}}{2^{12}}=\\frac{79}{4096}=0.0193",
          callout: "p=0.0193 means that, under H₀ and the sampling assumptions, the probability that at least 10 of 12 improve is about 1.93%."
        },
        {
          title: "5. Compare p with α and conclude carefully",
          body: ["Because 0.0193 < 0.05, reject H₀ at the prespecified 5% level.", "The data support a positive-change probability above 50%; this does not mean H₁ has probability 98.07%.", "Also report the raw proportion 10/12, its interval, the number of zero differences, and practical importance."],
          formula: "p=0.0193<\\alpha=0.05\\quad\\Longrightarrow\\quad \\text{reject }H_0",
          callout: "In one line: p locates this result in the tail of the H₀ reference distribution; α is the decision boundary drawn before seeing the data."
        }
      ]
    }
  };

  function text(value, lang) {
    if (value == null) return "";
    if (value && typeof value === "object" && Object.prototype.hasOwnProperty.call(value, lang)) return value[lang];
    return value;
  }
  function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>\"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }
  function getLang() {
    const urlLang = new URLSearchParams(location.search).get("lang");
    if (urlLang === "en" || urlLang === "zh") return urlLang;
    return localStorage.getItem("hypothesis-guide-lang") || "zh";
  }
  let LANG = getLang();
  function localized(value) { return text(value, LANG); }
  function safeUrl(path, params = {}) {
    const q = new URLSearchParams(params); q.set("lang", LANG);
    return `${BASE_PATH}${path}${q.toString() ? `?${q}` : ""}`;
  }
  function syncLanguage() {
    document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
    localStorage.setItem("hypothesis-guide-lang", LANG);
    document.querySelectorAll("[data-i18n]").forEach(node => { const value = UI_TEXT[LANG][node.dataset.i18n]; if (value != null) node.textContent = value; });
    document.querySelectorAll("[data-i18n-aria-label]").forEach(node => { const value = UI_TEXT[LANG][node.dataset.i18nAriaLabel]; if (value != null) node.setAttribute("aria-label", value); });
    const toggle = document.getElementById("language-toggle");
    if (toggle) { toggle.setAttribute("aria-label", UI_TEXT[LANG].switchLanguage); toggle.title = UI_TEXT[LANG].switchLanguage; toggle.querySelector(".lang-current").textContent = UI_TEXT[LANG].languageCode; }
    document.title = LANG === "zh" ? "假设检验指南｜方法选择" : "Hypothesis Testing Guide | Find a method";
    document.querySelectorAll("[data-nav]").forEach(node => {
      const nav = node.dataset.nav;
      node.href = nav === "help" ? safeUrl("help.html") : safeUrl("index.html", { view: "guide" });
    });
  }
  function rerenderMath(root = document.getElementById("app")) {
    if (!root) return;
    root.querySelectorAll("[data-tex]").forEach(node => {
      const source = node.dataset.tex || node.textContent || "";
      if (window.katex) {
        try { window.katex.render(source, node, { displayMode: true, throwOnError: false, strict: "warn" }); return; } catch (_) {}
      }
      node.textContent = source;
      node.classList.add("formula-fallback");
    });
    if (window.renderMathInElement) { try { window.renderMathInElement(root, { delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }], throwOnError: false }); } catch (_) {} }
  }
  function focusMain() { const heading = document.querySelector("#app h1, #app h2"); if (heading) { heading.setAttribute("tabindex", "-1"); heading.focus({ preventScroll: true }); } }
  function status(message) { const node = document.getElementById("page-status"); if (node) node.textContent = message; }
  function cardLink(id, label = null) { return `<a class="method-card" href="${safeUrl("method.html", { id })}" data-method-link="${id}"><span class="method-card-title">${escapeHTML(label || localized(METHODS[id]?.zh?.name || METHODS[id]?.en?.name || id))}</span><span class="method-card-meta">${escapeHTML(localized(METHODS[id]?.category || ""))}</span></a>`; }
  function methodDisplay(id) { return METHODS[id] ? { id, item: METHODS[id], copy: METHODS[id][LANG] } : CONCEPTS[id] ? { id, item: CONCEPTS[id], copy: CONCEPTS[id][LANG] } : null; }
  function allEntries() { return [...Object.values(METHODS), ...CONCEPT_ENTRIES]; }
  function normalizeSearch(value) {
    return String(value ?? "").normalize("NFKC").toLowerCase().replace(/[’']/g, "").replace(/[^\p{L}\p{N}]+/gu, " ").replace(/\s+/g, " ").trim();
  }
  function compactSearch(value) { return normalizeSearch(value).replace(/\s+/g, ""); }
  function searchFields(item) {
    const primary = [item.id, item.zh?.name, item.en?.name].filter(Boolean);
    const aliases = (item.aliases || []).filter(Boolean);
    const supporting = [
      item.family, item.category?.zh, item.category?.en,
      item.zh?.short, item.en?.short, item.zh?.background, item.en?.background,
      item.zh?.useWhen, item.en?.useWhen, item.zh?.hypotheses?.h0, item.zh?.hypotheses?.h1,
      item.en?.hypotheses?.h0, item.en?.hypotheses?.h1, item.python,
      ...(item.formulas || []).flatMap(value => [value.tex, value.label?.zh, value.label?.en]),
      ...(item.symbols || []).flatMap(value => [value.symbol, value.meaning?.zh, value.meaning?.en])
    ].filter(Boolean);
    return { primary, aliases, supporting };
  }
  function searchScore(item, query) {
    const q = normalizeSearch(query); const compactQ = compactSearch(query); if (!q) return 1;
    const fields = searchFields(item);
    const normalizedPrimary = fields.primary.map(normalizeSearch); const normalizedAliases = fields.aliases.map(normalizeSearch);
    const exact = values => values.some(value => value === q || compactSearch(value) === compactQ);
    if (exact(normalizedPrimary)) return 120;
    if (exact(normalizedAliases)) return 100;
    if (normalizedPrimary.some(value => value.startsWith(q) || compactSearch(value).startsWith(compactQ))) return 90;
    if (normalizedAliases.some(value => value.startsWith(q) || compactSearch(value).startsWith(compactQ))) return 80;
    const corpus = normalizeSearch([...fields.primary, ...fields.aliases, ...fields.supporting].join(" "));
    const corpusTokens = new Set(corpus.split(" ").filter(Boolean)); const queryTokens = q.split(" ").filter(Boolean);
    if (queryTokens.every(token => corpusTokens.has(token))) return 60;
    if (compactQ.length >= 2 && compactSearch(corpus).includes(compactQ)) return 40;
    if (queryTokens.every(token => token.length > 1 && corpus.includes(token))) return 20;
    return 0;
  }
  function searchEntries(query, filters = {}) {
    const candidates = allEntries().map((item, index) => ({ item, index })).filter(({ item }) => {
      if (filters.outcome && item.outcome !== filters.outcome) return false;
      if (filters.design && item.design !== filters.design) return false;
      if (filters.groups && item.groups !== filters.groups) return false;
      if (filters.kind && item.kind !== filters.kind) return false;
      return true;
    });
    if (!normalizeSearch(query)) return candidates.map(({ item }) => item);
    const scored = candidates.map(({ item, index }) => ({ item, index, score: searchScore(item, query) })).filter(value => value.score > 0);
    const hasExactMatch = scored.some(value => value.score >= 100);
    return scored.filter(value => !hasExactMatch || value.score >= 100).sort((a, b) => b.score - a.score || a.index - b.index).map(({ item }) => item);
  }
  function filtersHtml() {
    const options = (label, key, values) => `<label class="filter-field"><span>${escapeHTML(label)}</span><select data-filter="${key}"><option value="">${escapeHTML(UI_TEXT[LANG].any)}</option>${values.map(v => `<option value="${v}">${escapeHTML(UI_TEXT[LANG][v] || v)}</option>`).join("")}</select></label>`;
    return `<div class="filter-row">${options(UI_TEXT[LANG].filterOutcome, "outcome", ["continuous", "categorical", "ordinal", "count"])}${options(UI_TEXT[LANG].filterDesign, "design", ["one", "independent", "paired"])}${options(UI_TEXT[LANG].filterGroups, "groups", ["one", "two", "many"])}${options(UI_TEXT[LANG].filterKind, "kind", ["comparison", "association", "diagnostic", "effect"])}<button class="button ghost" type="button" data-clear-filters>${escapeHTML(UI_TEXT[LANG].clearFilters)}</button></div>`;
  }
  function searchBox(resultsId = "library-results") {
    return `<div class="search-panel"><label for="method-search">${escapeHTML(UI_TEXT[LANG].searchLabel)}</label><div class="search-line"><input id="method-search" type="search" autocomplete="off" placeholder="${escapeHTML(UI_TEXT[LANG].searchPlaceholder)}" aria-describedby="search-hint" aria-controls="${escapeHTML(resultsId)}"><button class="button ghost" type="button" data-clear-search>${escapeHTML(UI_TEXT[LANG].clearSearch)}</button></div><p class="search-hint" id="search-hint">${escapeHTML(UI_TEXT[LANG].searchHint)}</p></div>`;
  }
  function renderResults(container, query = "", filters = {}) {
    const entries = searchEntries(query, filters);
    const statusNode = document.getElementById("library-results-status");
    if (statusNode) statusNode.textContent = `${entries.length} ${UI_TEXT[LANG].resultCount}`;
    if (!entries.length) { container.innerHTML = `<p class="empty-state">${escapeHTML(UI_TEXT[LANG].noResults)}</p>`; return; }
    const methods = entries.filter(x => x.kind !== "effect"); const effects = entries.filter(x => x.kind === "effect");
    const group = (title, items) => items.length ? `<section class="result-group"><h3>${escapeHTML(title)}</h3><div class="method-grid">${items.map(item => cardLink(item.id, localized(item[LANG]?.name || item.zh?.name))).join("")}</div></section>` : "";
    container.innerHTML = group(UI_TEXT[LANG].methods, methods) + group(UI_TEXT[LANG].concepts, effects);
  }
  function bindSearch(root) {
    const input = root.querySelector("#method-search"); const results = root.querySelector("#library-results"); if (!input || !results) return;
    const render = () => { const filters = Object.fromEntries([...root.querySelectorAll("[data-filter]")].map(s => [s.dataset.filter, s.value])); renderResults(results, input.value, filters); };
    input.addEventListener("input", render); root.querySelectorAll("[data-filter]").forEach(s => s.addEventListener("change", render));
    root.querySelector("[data-clear-search]")?.addEventListener("click", () => { input.value = ""; render(); input.focus(); });
    root.querySelector("[data-clear-filters]")?.addEventListener("click", () => { root.querySelectorAll("[data-filter]").forEach(s => s.value = ""); render(); });
    render();
  }
  function renderDecisionTreeNode(nodeId, depth = 0) {
    const node = DECISION_TREE[nodeId];
    if (!node || depth > 8) return "";
    const branches = node.options.map(option => {
      const destination = option.result ? METHODS[option.result] : null;
      const child = option.next ? renderDecisionTreeNode(option.next, depth + 1) : "";
      const methodName = destination ? (destination[LANG]?.name || destination.zh?.name || destination.en?.name || option.result) : ""; const methodCategory = destination ? localized(destination.category) : "";
      const leaf = destination ? `<a class="decision-method-leaf" href="${safeUrl("method.html", { id: option.result })}" aria-label="${escapeHTML(`${methodName} — ${UI_TEXT[LANG].openMethodShort}`)}"><strong class="decision-method-name">${escapeHTML(methodName)}</strong><small class="decision-method-meta">${escapeHTML(methodCategory)}</small><span class="decision-method-action">${escapeHTML(UI_TEXT[LANG].openMethodShort)} <b aria-hidden="true">→</b></span></a>` : "";
      const advice = option.note ? `<div class="decision-advice-leaf" role="note"><strong>${escapeHTML(UI_TEXT[LANG].alternativeMethod)}</strong><span>${escapeHTML(localized(option.note))}</span></div>` : "";
      const branchType = child ? "has-child" : destination ? "has-leaf" : advice ? "has-advice" : "has-condition";
      return `<li class="decision-branch ${branchType}"><div class="decision-condition"><strong>${escapeHTML(localized(option.label))}</strong><span>${escapeHTML(localized(option.desc))}</span></div>${leaf}${advice}${child}</li>`;
    }).join("");
    return `<details class="decision-tree-node depth-${depth}" open><summary><span>${escapeHTML(localized(node.title))}</span></summary><ul>${branches}</ul></details>`;
  }
  function indexModeHeader(activeMode) {
    const guideActive = activeMode === "guide"; const libraryActive = activeMode === "library";
    const modeCard = (mode, title, description, active, symbol) => `<a class="find-mode-card${active ? " is-active" : ""}" href="${safeUrl("index.html", { view: mode })}"${active ? ` aria-current="page"` : ""}><span class="find-mode-symbol" aria-hidden="true">${symbol}</span><span class="find-mode-copy">${active ? `<span class="find-mode-status">${escapeHTML(UI_TEXT[LANG].currentMode)}</span>` : ""}<strong>${escapeHTML(title)}</strong><small>${escapeHTML(description)}</small></span></a>`;
    return `<section class="hero-block compact index-mode-hero"><p class="eyebrow">${escapeHTML(LANG === "zh" ? "方法选择" : "Method finder")}</p><h1>${escapeHTML(UI_TEXT[LANG].findMethodTitle)}</h1><p class="lead">${escapeHTML(UI_TEXT[LANG].findMethodLead)}</p></section><nav class="find-mode-switch" aria-label="${escapeHTML(UI_TEXT[LANG].modeLabel)}">${modeCard("guide", UI_TEXT[LANG].guideModeTitle, UI_TEXT[LANG].guideModeDesc, guideActive, "↳")}${modeCard("library", UI_TEXT[LANG].libraryModeTitle, UI_TEXT[LANG].libraryModeDesc, libraryActive, "⌕")}</nav>`;
  }
  function renderFinder(root) {
    const intro = LANG === "zh"
      ? "从研究目标开始沿树向下读。所有分支和方法默认展开；条件是选择依据，绿色叶节点可直接打开完整公式与例题。"
      : "Start from the research goal and read down the tree. Every branch and method is expanded; conditions guide the choice, and green leaves open the full formulas and example.";
    const expand = LANG === "zh" ? "展开全部" : "Expand all";
    const collapse = LANG === "zh" ? "收起分支" : "Collapse branches";
    root.innerHTML = `${indexModeHeader("guide")}<section class="decision-tree-shell" aria-labelledby="decision-tree-title"><div class="decision-tree-toolbar"><h2 id="decision-tree-title">${escapeHTML(LANG === "zh" ? "完整检验选择树" : "Complete test-selection tree")}</h2><div><button class="button ghost tiny" type="button" data-tree-expand>${escapeHTML(expand)}</button><button class="button ghost tiny" type="button" data-tree-collapse>${escapeHTML(collapse)}</button></div></div><p class="decision-tree-intro">${escapeHTML(intro)}</p><div class="decision-tree" aria-label="${escapeHTML(LANG === "zh" ? "从研究目标到统计方法的完整树" : "Complete tree from research goals to statistical methods")}">${renderDecisionTreeNode("start")}</div><p class="hint">${escapeHTML(LANG === "zh" ? "同一方法可能出现在多个合理路径中；树是选择提示，不替代研究设计审查。" : "A method can appear on several valid paths. The tree guides selection but does not replace design review.")}</p></section>`;
    root.querySelector("[data-tree-expand]")?.addEventListener("click", () => root.querySelectorAll(".decision-tree-node").forEach(node => { node.open = true; }));
    root.querySelector("[data-tree-collapse]")?.addEventListener("click", () => root.querySelectorAll(".decision-tree-node").forEach((node, index) => { node.open = index === 0; }));
  }
  function renderLibrary(root) {
    root.innerHTML = `${indexModeHeader("library")}<section class="library-mode-intro" aria-labelledby="library-mode-title"><h2 id="library-mode-title">${escapeHTML(UI_TEXT[LANG].methodLibrary)}</h2><p>${escapeHTML(UI_TEXT[LANG].libraryModeDesc)}</p></section>${searchBox("library-results")}${filtersHtml()}<p id="library-results-status" class="sr-only" aria-live="polite"></p><div id="library-results"></div>`;
    bindSearch(root);
  }
  function renderIndex() {
    const root = document.getElementById("app"); if (!root) return;
    const view = new URLSearchParams(location.search).get("view") || "guide";
    document.querySelectorAll("[data-nav]").forEach(n => n.removeAttribute("aria-current")); document.querySelector('[data-nav="finder"]')?.setAttribute("aria-current", "page");
    if (view === "library") { document.title = LANG === "zh" ? "搜索方法库｜假设检验指南" : "Search the method library | Hypothesis Testing Guide"; renderLibrary(root); if (new URLSearchParams(location.search).get("focus") === "search") requestAnimationFrame(() => root.querySelector("#method-search")?.focus()); } else { document.title = LANG === "zh" ? "沿选择树判断｜假设检验指南" : "Follow the decision tree | Hypothesis Testing Guide"; root.innerHTML = `<div id="finder-root"></div>`; renderFinder(root.querySelector("#finder-root")); }
    rerenderMath();
  }
  function tableHtml(example) {
    if (!example) return ""; const cols = localized(example.columns); return `<div class="table-wrap"><table><caption>${escapeHTML(localized(example.caption))}</caption><thead><tr>${cols.map(c => `<th scope="col">${escapeHTML(c)}</th>`).join("")}</tr></thead><tbody>${example.rows.map(row => `<tr>${row.map(cell => `<td>${escapeHTML(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  function listHtml(value) { const values = Array.isArray(value) ? value : [value]; return `<ul class="check-list">${values.map(x => `<li>${escapeHTML(x)}</li>`).join("")}</ul>`; }
  function formulaHtml(formulas) { return `<div class="formula-list">${(formulas || []).map(f => `<div class="formula-card"><h3>${escapeHTML(localized(f.label))}</h3><div class="formula" data-tex="${escapeHTML(f.tex)}">${escapeHTML(f.tex)}</div></div>`).join("")}</div>`; }
  function symbolsHtml(symbols) { return `<div class="symbol-grid">${(symbols || []).map(s => `<div><code>${escapeHTML(s.symbol)}</code><span>${escapeHTML(localized(s.meaning))}</span></div>`).join("")}</div>`; }
  function renderCode(code) { return code ? `<div class="code-block"><div class="code-toolbar"><span>Python</span><button class="button tiny" type="button" data-copy-code>${escapeHTML(UI_TEXT[LANG].copy)}</button></div><pre><code>${escapeHTML(code)}</code></pre></div>` : ""; }
  function renderMethod() {
    const root = document.getElementById("app"); if (!root) return;
    const id = new URLSearchParams(location.search).get("id") || "paired_t"; const found = methodDisplay(id);
    if (!found) { document.title = LANG === "zh" ? "方法不存在｜假设检验指南" : "Method not found | Hypothesis Testing Guide"; root.innerHTML = `<section class="empty-state"><h1>${escapeHTML(UI_TEXT[LANG].notFound)}</h1><p>${escapeHTML(UI_TEXT[LANG].notFoundText)}</p><a class="button primary" href="${safeUrl("index.html", { view: "library" })}">${escapeHTML(UI_TEXT[LANG].backToLibrary)}</a></section>`; return; }
    const { item, copy } = found; const related = (item.related || []).filter(x => METHODS[x] || CONCEPTS[x]);
    document.title = LANG === "zh" ? `${copy.name}｜假设检验指南` : `${copy.name} | Hypothesis Testing Guide`;
    root.innerHTML = `<div class="detail-layout"><aside class="toc panel"><p class="eyebrow">${escapeHTML(localized(item.category))}</p><h2>${escapeHTML(UI_TEXT[LANG].contents)}</h2><nav><a href="#background">${escapeHTML(UI_TEXT[LANG].background)}</a><a href="#formula">${escapeHTML(UI_TEXT[LANG].formula)}</a><a href="#example">${escapeHTML(UI_TEXT[LANG].example)}</a><a href="#inference">${escapeHTML(UI_TEXT[LANG].inference)}</a><a href="#python">${escapeHTML(UI_TEXT[LANG].python)}</a><a href="#reporting">${escapeHTML(UI_TEXT[LANG].reporting)}</a></nav></aside><article class="method-article"><p class="breadcrumbs"><a href="${safeUrl("index.html", { view: "library" })}">${escapeHTML(UI_TEXT[LANG].methodLibrary)}</a><span aria-hidden="true">/</span><span>${escapeHTML(localized(item.category))}</span></p><header class="method-header"><span class="pill">${escapeHTML(localized(item.category))}</span><h1>${escapeHTML(copy.name)}</h1><p class="lead">${escapeHTML(copy.short)}</p></header><section id="background" class="content-section"><h2>${escapeHTML(UI_TEXT[LANG].background)}</h2><p>${escapeHTML(copy.background)}</p><div class="two-col"><div class="info-card"><h3>${escapeHTML(UI_TEXT[LANG].useWhen)}</h3><p>${escapeHTML(copy.useWhen)}</p></div><div class="info-card"><h3>${escapeHTML(UI_TEXT[LANG].avoidWhen)}</h3><p>${escapeHTML(copy.avoidWhen)}</p></div></div></section><section class="content-section"><h2>${escapeHTML(UI_TEXT[LANG].hypotheses)}</h2><div class="two-col"><div class="hypothesis h0"><h3>${escapeHTML(UI_TEXT[LANG].nullHypothesis)}</h3><p>${escapeHTML(copy.hypotheses.h0)}</p></div><div class="hypothesis h1"><h3>${escapeHTML(UI_TEXT[LANG].alternativeHypothesis)}</h3><p>${escapeHTML(copy.hypotheses.h1)}</p></div></div></section><section class="content-section"><h2>${escapeHTML(UI_TEXT[LANG].assumptions)}</h2>${listHtml(copy.assumptions)}</section><section id="formula" class="content-section"><h2>${escapeHTML(UI_TEXT[LANG].formula)}</h2>${symbolsHtml(item.symbols)}${formulaHtml(item.formulas)}<p class="hint">${escapeHTML(copy.formulaNotes)}</p></section><section id="example" class="content-section"><h2>${escapeHTML(UI_TEXT[LANG].example)}</h2>${tableHtml(item.example)}<ol class="calculation-steps">${(localized(item.example?.steps) || []).map(step => `<li>${escapeHTML(step)}</li>`).join("")}</ol><div class="result-callout"><strong>${escapeHTML(UI_TEXT[LANG].result)}</strong><p>${escapeHTML(localized(item.example?.result))}</p></div></section><section id="inference" class="content-section"><h2>${escapeHTML(UI_TEXT[LANG].inference)}</h2><p>${escapeHTML(copy.inference)}</p><p>${escapeHTML(copy.ci)}</p><p>${escapeHTML(copy.effect)}</p></section><section class="content-section"><h2>${escapeHTML(UI_TEXT[LANG].edgeCases)}</h2>${listHtml(copy.edgeCases)}</section><section id="reporting" class="content-section"><h2>${escapeHTML(UI_TEXT[LANG].reporting)}</h2><div class="report-box"><p>${escapeHTML(copy.report)}</p><button class="button tiny" type="button" data-copy-report>${escapeHTML(UI_TEXT[LANG].copy)}</button></div></section><section id="python" class="content-section"><h2>${escapeHTML(UI_TEXT[LANG].python)}</h2>${renderCode(item.python)}</section><section class="content-section"><h2>${escapeHTML(UI_TEXT[LANG].related)}</h2><div class="method-grid">${related.map(r => cardLink(r, localized((METHODS[r] || CONCEPTS[r])[LANG]?.name || (METHODS[r] || CONCEPTS[r]).zh?.name))).join("")}</div></section><section class="content-section"><h2>${escapeHTML(UI_TEXT[LANG].references)}</h2><ul class="reference-list">${(item.references || []).map(r => `<li><a href="${r.url}" target="_blank" rel="noreferrer">${escapeHTML(localized(r.label))}</a></li>`).join("")}</ul></section></article></div>`;
    root.querySelectorAll("[data-copy-code]").forEach(btn => btn.addEventListener("click", () => copyText(item.python, btn)));
    root.querySelector("[data-copy-report]")?.addEventListener("click", () => copyText(copy.report, root.querySelector("[data-copy-report]")));
    rerenderMath();
  }
  const P_ALPHA_STATE = { distribution: "normal", alternative: "two", observed: 2, alpha: 0.05, run: 0, timer: null, storyStep: 0 };
  function clamp01(value) { return Math.max(0, Math.min(1, value)); }
  function normalPdf(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
  function erfApprox(value) {
    const sign = value < 0 ? -1 : 1; const x = Math.abs(value); const t = 1 / (1 + 0.3275911 * x);
    const poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign * (1 - poly * Math.exp(-x * x));
  }
  function normalCdf(x) { return clamp01(0.5 * (1 + erfApprox(x / Math.sqrt(2)))); }
  function studentPdf5(x) { return (8 / (3 * Math.PI * Math.sqrt(5))) * Math.pow(1 + x * x / 5, -3); }
  function studentCdf5(x) {
    if (x === 0) return 0.5;
    const upper = Math.min(Math.abs(x), 40); const n = 400; const h = upper / n;
    let sum = studentPdf5(0) + studentPdf5(upper);
    for (let i = 1; i < n; i += 1) sum += (i % 2 ? 4 : 2) * studentPdf5(i * h);
    const area = sum * h / 3; return clamp01(x > 0 ? 0.5 + area : 0.5 - area);
  }
  function chiSquarePdf4(x) { return x <= 0 ? 0 : (x / 4) * Math.exp(-x / 2); }
  function chiSquareCdf4(x) { return x <= 0 ? 0 : clamp01(1 - Math.exp(-x / 2) * (1 + x / 2)); }
  function choose(n, k) {
    if (k < 0 || k > n) return 0; let answer = 1;
    for (let i = 1; i <= Math.min(k, n - k); i += 1) answer = answer * (n - i + 1) / i;
    return answer;
  }
  function binomialPmf12(k) { return choose(12, k) / 4096; }
  function pAlphaConfig(id) {
    return {
      normal: { min: -4, max: 4, step: 0.1, initial: 2, pdf: normalPdf, cdf: normalCdf },
      student: { min: -5, max: 5, step: 0.1, initial: 2, pdf: studentPdf5, cdf: studentCdf5 },
      chiSquare: { min: 0, max: 16, step: 0.1, initial: 8, pdf: chiSquarePdf4, cdf: chiSquareCdf4 },
      binomial: { min: 0, max: 12, step: 1, initial: 9, discrete: true }
    }[id];
  }
  function labPValue(distribution, observed, alternative) {
    if (distribution === "binomial") {
      const k = Math.round(observed); const probs = Array.from({ length: 13 }, (_, i) => binomialPmf12(i));
      if (alternative === "greater") return probs.slice(k).reduce((a, b) => a + b, 0);
      if (alternative === "less") return probs.slice(0, k + 1).reduce((a, b) => a + b, 0);
      const threshold = probs[k] + 1e-12; return clamp01(probs.filter(prob => prob <= threshold).reduce((a, b) => a + b, 0));
    }
    const config = pAlphaConfig(distribution); const cdf = config.cdf(observed);
    if (distribution === "chiSquare" || alternative === "greater") return clamp01(1 - cdf);
    if (alternative === "less") return cdf;
    return clamp01(2 * Math.min(cdf, 1 - cdf));
  }
  function pAlphaFormula(distribution, alternative) {
    if (distribution === "binomial") {
      if (alternative === "greater") return "p=\\sum_{j=k_{obs}}^{12}{12 \\choose j}(0.5)^{12}";
      if (alternative === "less") return "p=\\sum_{j=0}^{k_{obs}}{12 \\choose j}(0.5)^{12}";
      return "p=\\sum_{j:\\,P_0(K=j)\\le P_0(K=k_{obs})}P_0(K=j)";
    }
    const symbol = distribution === "chiSquare" ? "\\chi^2" : distribution === "student" ? "T" : "Z";
    if (distribution === "chiSquare" || alternative === "greater") return `p=P_0(${symbol}\\ge ${symbol}_{obs})=1-F_0(${symbol}_{obs})`;
    if (alternative === "less") return `p=P_0(${symbol}\\le ${symbol}_{obs})=F_0(${symbol}_{obs})`;
    return `p=P_0(|${symbol}|\\ge|${symbol}_{obs}|)=2\\min\\{F_0(${symbol}_{obs}),1-F_0(${symbol}_{obs})\\}`;
  }
  function formatProbability(value) { return value < 0.0001 ? "< 0.0001" : value.toFixed(4); }
  function chartAreaPath(points, predicate, xScale, yScale, baseline) {
    const groups = []; let group = [];
    points.forEach(point => { if (predicate(point.x)) group.push(point); else if (group.length) { groups.push(group); group = []; } });
    if (group.length) groups.push(group);
    return groups.filter(items => items.length > 1).map(items => `M ${xScale(items[0].x).toFixed(2)} ${baseline} L ${items.map(point => `${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`).join(" L ")} L ${xScale(items.at(-1).x).toFixed(2)} ${baseline} Z`).join(" ");
  }
  function continuousNullChart(distribution, observed, alternative, alpha, copy, idPrefix = "lab") {
    const config = pAlphaConfig(distribution); const width = 760; const height = 280; const left = 48; const right = 24; const top = 20; const baseline = 226;
    const points = Array.from({ length: 241 }, (_, i) => { const x = config.min + (config.max - config.min) * i / 240; return { x, y: config.pdf(x) }; });
    const maxY = Math.max(...points.map(point => point.y)) * 1.12; const xScale = x => left + (x - config.min) / (config.max - config.min) * (width - left - right); const yScale = y => baseline - y / maxY * (baseline - top);
    const pExtreme = x => alternative === "greater" || distribution === "chiSquare" ? x >= observed : alternative === "less" ? x <= observed : Math.abs(x) >= Math.abs(observed);
    const alphaReject = x => labPValue(distribution, x, alternative) <= alpha;
    const alphaPath = chartAreaPath(points, alphaReject, xScale, yScale, baseline); const pPath = chartAreaPath(points, pExtreme, xScale, yScale, baseline);
    const curvePath = points.map((point, index) => `${index ? "L" : "M"} ${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`).join(" ");
    const ticks = Array.from({ length: 5 }, (_, i) => config.min + (config.max - config.min) * i / 4);
    const titleId = `${idPrefix}-chart-title`; const descId = `${idPrefix}-chart-desc`; const hatchId = `${idPrefix}-alpha-hatch`;
    return `<svg class="lab-chart" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${titleId} ${descId}"><title id="${titleId}">${escapeHTML(copy.chartLabel)}</title><desc id="${descId}">${escapeHTML(copy.pMeaning)} ${escapeHTML(copy.alphaMeaning)}</desc><defs><pattern id="${hatchId}" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line class="alpha-hatch-line" x1="0" y1="0" x2="0" y2="8" /></pattern></defs><line class="lab-axis" x1="${left}" y1="${baseline}" x2="${width - right}" y2="${baseline}" />${alphaPath ? `<path class="lab-alpha-area" style="fill:url(#${hatchId})" d="${alphaPath}" />` : ""}${pPath ? `<path class="lab-p-area" d="${pPath}" />` : ""}<path class="lab-density-line" d="${curvePath}" /><line class="lab-observed-line" x1="${xScale(observed)}" y1="${top}" x2="${xScale(observed)}" y2="${baseline}" /><text class="lab-observed-text" x="${Math.min(width - 90, Math.max(left + 8, xScale(observed) + 7))}" y="${top + 16}">${escapeHTML(copy.observedLabel)} ${Number(observed).toFixed(1)}</text>${ticks.map(tick => `<g><line class="lab-tick" x1="${xScale(tick)}" y1="${baseline}" x2="${xScale(tick)}" y2="${baseline + 6}"/><text class="lab-tick-label" x="${xScale(tick)}" y="${baseline + 24}">${Number(tick).toFixed(tick % 1 ? 1 : 0)}</text></g>`).join("")}</svg>`;
  }
  function discreteNullChart(observed, alternative, alpha, copy, idPrefix = "lab") {
    const width = 760; const height = 280; const left = 42; const right = 22; const baseline = 226; const top = 24; const probs = Array.from({ length: 13 }, (_, k) => binomialPmf12(k)); const maxP = Math.max(...probs) * 1.12; const slot = (width - left - right) / 13; const barWidth = slot * 0.62; const obsP = probs[Math.round(observed)];
    const bars = probs.map((prob, k) => {
      const x = left + k * slot + (slot - barWidth) / 2; const barHeight = prob / maxP * (baseline - top); const inP = alternative === "greater" ? k >= observed : alternative === "less" ? k <= observed : prob <= obsP + 1e-12; const rejected = labPValue("binomial", k, alternative) <= alpha;
      return `<g><rect class="lab-discrete-bar${inP ? " in-p" : ""}${rejected ? " is-rejection" : ""}${k === Math.round(observed) ? " is-observed" : ""}" x="${x}" y="${baseline - barHeight}" width="${barWidth}" height="${barHeight}" /><text class="lab-tick-label" x="${x + barWidth / 2}" y="${baseline + 22}">${k}</text></g>`;
    }).join("");
    const titleId = `${idPrefix}-chart-title`; const descId = `${idPrefix}-chart-desc`;
    return `<svg class="lab-chart" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${titleId} ${descId}"><title id="${titleId}">${escapeHTML(copy.chartLabel)}</title><desc id="${descId}">${escapeHTML(copy.discreteMeaning)}</desc><line class="lab-axis" x1="${left}" y1="${baseline}" x2="${width - right}" y2="${baseline}" />${bars}<text class="lab-observed-text" x="${left + Math.round(observed) * slot + slot / 2}" y="${top}">${escapeHTML(copy.observedLabel)} k=${Math.round(observed)}</text></svg>`;
  }
  function storyVisual(index, step) {
    if (index === 0) {
      const labels = LANG === "zh" ? ["同一人前后测量", "得到配对差值", "这里只记录 + / −", "问题：正差是否 > 50%"] : ["Before–after pairs", "Compute paired differences", "Record only + / −", "Question: positives > 50%?"];
      return `<div class="story-flow" aria-label="${escapeHTML(labels.join(" → "))}">${labels.map((label, i) => `<span>${escapeHTML(label)}</span>${i < labels.length - 1 ? `<b aria-hidden="true">→</b>` : ""}`).join("")}</div>`;
    }
    if (index === 2) {
      return `<div class="story-alpha-visual"><div class="story-alpha-grid" aria-label="${escapeHTML(LANG === "zh" ? "100 次长期重复中，用 5 个标记表示名义 5% 错误水平" : "Five marked cells among 100 illustrate a nominal 5% long-run error level")}">${Array.from({ length: 100 }, (_, i) => `<i class="${i < 5 ? "is-alpha" : ""}" aria-hidden="true"></i>`).join("")}</div><div class="formula" data-tex="${escapeHTML(step.formula)}">${escapeHTML(step.formula)}</div></div>`;
    }
    if (index === 3) {
      const outcomes = Array.from({ length: 12 }, (_, i) => i < 10 ? "+" : "−");
      return `<div class="story-observations" aria-label="${escapeHTML(LANG === "zh" ? "10 个正差和 2 个负差" : "10 positive and 2 negative differences")}">${outcomes.map(value => `<span class="${value === "+" ? "is-positive" : "is-negative"}">${value}</span>`).join("")}</div><div class="story-binomial-chart">${discreteNullChart(10, "greater", 0.05, P_ALPHA_TEXT[LANG], "story")}</div><div class="formula" data-tex="${escapeHTML(step.formula)}">${escapeHTML(step.formula)}</div>`;
    }
    if (index === 4) {
      return `<div class="story-comparison"><span>p = 0.0193</span><b aria-hidden="true">&lt;</b><span>α = 0.05</span></div><div class="formula" data-tex="${escapeHTML(step.formula)}">${escapeHTML(step.formula)}</div>`;
    }
    return `<div class="story-hypotheses"><div class="formula" data-tex="${escapeHTML(step.formula)}">${escapeHTML(step.formula)}</div></div>`;
  }
  function renderPAlphaStory() {
    const story = P_ALPHA_STORY[LANG]; const count = story.steps.length;
    return `<section class="p-alpha-story" id="p-alpha-start" aria-labelledby="p-alpha-story-title"><header><p class="eyebrow">${escapeHTML(story.eyebrow)}</p><h2 id="p-alpha-story-title">${escapeHTML(story.title)}</h2><p class="lead">${escapeHTML(story.lead)}</p></header><nav class="story-step-nav" aria-label="${escapeHTML(story.stepLabel)}">${story.steps.map((step, index) => `<button type="button" data-story-step="${index}" aria-pressed="${index === P_ALPHA_STATE.storyStep}"><span>${index + 1}</span><small>${escapeHTML(step.title.replace(/^\d+\.\s*/, ""))}</small></button>`).join("")}</nav><div class="story-stage" id="story-stage">${story.steps.map((step, index) => `<article class="story-panel" data-story-panel="${index}"${index === P_ALPHA_STATE.storyStep ? "" : " hidden"}><div class="story-copy"><p class="story-counter">${escapeHTML(story.stepLabel)} ${index + 1} / ${count}</p><h3>${escapeHTML(step.title)}</h3>${listHtml(step.body)}<div class="story-callout">${escapeHTML(step.callout)}</div></div><div class="story-visual">${storyVisual(index, step)}</div></article>`).join("")}</div><div class="story-actions"><button class="button secondary" type="button" data-story-previous>${escapeHTML(story.previous)}</button><span id="story-progress" aria-live="polite">${P_ALPHA_STATE.storyStep + 1} / ${count}</span><button class="button primary" type="button" data-story-next>${escapeHTML(P_ALPHA_STATE.storyStep === count - 1 ? story.restart : story.next)}</button></div></section>`;
  }
  function bindPAlphaStory(root) {
    const story = P_ALPHA_STORY[LANG]; const buttons = [...root.querySelectorAll("[data-story-step]")]; const panels = [...root.querySelectorAll("[data-story-panel]")]; const previous = root.querySelector("[data-story-previous]"); const next = root.querySelector("[data-story-next]"); const progress = root.querySelector("#story-progress");
    if (!buttons.length || !panels.length || !previous || !next || !progress) return;
    const draw = () => {
      buttons.forEach((button, index) => button.setAttribute("aria-pressed", String(index === P_ALPHA_STATE.storyStep)));
      panels.forEach((panel, index) => { panel.hidden = index !== P_ALPHA_STATE.storyStep; });
      previous.disabled = P_ALPHA_STATE.storyStep === 0; progress.textContent = `${P_ALPHA_STATE.storyStep + 1} / ${story.steps.length}`; next.textContent = P_ALPHA_STATE.storyStep === story.steps.length - 1 ? story.restart : story.next;
    };
    buttons.forEach((button, index) => button.addEventListener("click", () => { P_ALPHA_STATE.storyStep = index; draw(); }));
    previous.addEventListener("click", () => { P_ALPHA_STATE.storyStep = Math.max(0, P_ALPHA_STATE.storyStep - 1); draw(); });
    next.addEventListener("click", () => { P_ALPHA_STATE.storyStep = P_ALPHA_STATE.storyStep === story.steps.length - 1 ? 0 : P_ALPHA_STATE.storyStep + 1; draw(); });
    draw();
  }
  function renderPAlphaLab() {
    const copy = P_ALPHA_TEXT[LANG]; const config = pAlphaConfig(P_ALPHA_STATE.distribution); const observedLabel = P_ALPHA_STATE.distribution === "binomial" ? copy.successes : copy.observed;
    if (P_ALPHA_STATE.timer) { clearInterval(P_ALPHA_STATE.timer); P_ALPHA_STATE.timer = null; }
    return `<section class="p-alpha-topic" id="p-alpha"><header class="p-alpha-header"><p class="eyebrow">${escapeHTML(copy.eyebrow)}</p><h2>${escapeHTML(copy.title)}</h2><p class="lead">${escapeHTML(copy.lead)}</p></header><div class="p-alpha-lab"><div class="lab-controls"><label class="lab-control"><span>${escapeHTML(copy.distribution)}</span><select id="lab-distribution"><option value="normal">${escapeHTML(copy.normal)}</option><option value="student">${escapeHTML(copy.student)}</option><option value="chiSquare">${escapeHTML(copy.chiSquare)}</option><option value="binomial">${escapeHTML(copy.binomial)}</option></select></label><label class="lab-control"><span>${escapeHTML(copy.alternative)}</span><select id="lab-alternative"><option value="two">${escapeHTML(copy.twoSided)}</option><option value="greater">${escapeHTML(copy.greater)}</option><option value="less">${escapeHTML(copy.less)}</option></select></label><label class="lab-control range-control"><span><span id="lab-observed-label">${escapeHTML(observedLabel)}</span> <output id="lab-observed-output">${P_ALPHA_STATE.observed}</output></span><input id="lab-observed" type="range" min="${config.min}" max="${config.max}" step="${config.step}" value="${P_ALPHA_STATE.observed}"></label><label class="lab-control range-control"><span>${escapeHTML(copy.alpha)} <output id="lab-alpha-output">${P_ALPHA_STATE.alpha.toFixed(3)}</output></span><input id="lab-alpha" type="range" min="0.001" max="0.200" step="0.001" value="${P_ALPHA_STATE.alpha}"></label></div><div class="lab-readout"><div><span>${escapeHTML(copy.pValue)}</span><strong id="lab-p-output">—</strong></div><div><span>${escapeHTML(copy.alpha)}</span><strong id="lab-alpha-card">${P_ALPHA_STATE.alpha.toFixed(3)}</strong></div><div><span>${escapeHTML(copy.decision)}</span><strong id="lab-decision" aria-live="polite">—</strong></div></div><figure class="lab-figure"><div id="lab-chart-container"></div><figcaption><span class="lab-legend-item p-region"><i aria-hidden="true"></i>${escapeHTML(copy.pRegionLabel)}</span><span class="lab-legend-item alpha-region"><i aria-hidden="true"></i>${escapeHTML(copy.rejectionLabel)}</span></figcaption></figure><div class="lab-explanation"><div class="formula" id="lab-p-formula" data-tex=""></div><p id="lab-p-note"></p><p>${escapeHTML(copy.alphaMeaning)}</p></div><section class="lab-concepts" aria-labelledby="lab-concepts-title"><h3 id="lab-concepts-title">${escapeHTML(copy.relationTitle)}</h3><ol><li>${escapeHTML(copy.nullDistribution)}</li><li>${escapeHTML(copy.pDefinition)}</li><li>${escapeHTML(copy.alphaDefinition)}</li></ol><div class="result-callout"><strong>${escapeHTML(copy.interpretation)}</strong><p>${escapeHTML(copy.interpretationText)}</p></div></section><section class="lab-simulation" aria-labelledby="lab-simulation-title"><div><h3 id="lab-simulation-title">${escapeHTML(copy.simulateTitle)}</h3><p>${escapeHTML(copy.simulateLead)}</p></div><button class="button primary" type="button" id="lab-run">${escapeHTML(copy.run)}</button><div class="sim-summary" id="lab-sim-summary" aria-live="polite">${escapeHTML(copy.noneYet)}</div><div class="sim-grid" id="lab-sim-grid" role="img" aria-label="${escapeHTML(copy.simulationLabel)}"></div></section></div><div class="two-col p-alpha-reading"><section class="help-card"><h3>${escapeHTML(copy.notAllNormal)}</h3>${listHtml(copy.familyMap)}</section><section class="help-card"><h3>${escapeHTML(copy.cautions)}</h3>${listHtml(copy.cautionItems)}</section></div></section>`;
  }
  function seededRandom(seed) { let state = seed >>> 0; return () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; }; }
  function sampleNormal(random) { const u = Math.max(random(), 1e-12); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random()); }
  function sampleNull(distribution, random) {
    if (distribution === "normal") return sampleNormal(random);
    if (distribution === "student") { const z = sampleNormal(random); let sum = 0; for (let i = 0; i < 5; i += 1) { const value = sampleNormal(random); sum += value * value; } return z / Math.sqrt(sum / 5); }
    if (distribution === "chiSquare") { let sum = 0; for (let i = 0; i < 4; i += 1) { const value = sampleNormal(random); sum += value * value; } return sum; }
    let successes = 0; for (let i = 0; i < 12; i += 1) if (random() < 0.5) successes += 1; return successes;
  }
  function theoreticalRejectionRate(distribution, alternative, alpha) {
    if (distribution !== "binomial") return alpha;
    return Array.from({ length: 13 }, (_, k) => labPValue("binomial", k, alternative) <= alpha ? binomialPmf12(k) : 0).reduce((a, b) => a + b, 0);
  }
  function bindPAlphaLab(root) {
    const distribution = root.querySelector("#lab-distribution"); const alternative = root.querySelector("#lab-alternative"); const observed = root.querySelector("#lab-observed"); const alpha = root.querySelector("#lab-alpha"); const runButton = root.querySelector("#lab-run");
    if (!distribution || !alternative || !observed || !alpha || !runButton) return;
    distribution.value = P_ALPHA_STATE.distribution; alternative.value = P_ALPHA_STATE.alternative;
    const stopSimulation = () => { if (P_ALPHA_STATE.timer) { clearInterval(P_ALPHA_STATE.timer); P_ALPHA_STATE.timer = null; } };
    const resetSimulation = () => { stopSimulation(); root.querySelector("#lab-sim-grid").innerHTML = ""; root.querySelector("#lab-sim-summary").textContent = P_ALPHA_TEXT[LANG].noneYet; runButton.disabled = false; runButton.textContent = P_ALPHA_TEXT[LANG].run; };
    const update = (reset = true) => {
      const copy = P_ALPHA_TEXT[LANG]; const config = pAlphaConfig(P_ALPHA_STATE.distribution); const p = labPValue(P_ALPHA_STATE.distribution, P_ALPHA_STATE.observed, P_ALPHA_STATE.alternative); const rejected = p <= P_ALPHA_STATE.alpha;
      observed.min = config.min; observed.max = config.max; observed.step = config.step; observed.value = P_ALPHA_STATE.observed;
      alternative.disabled = P_ALPHA_STATE.distribution === "chiSquare";
      root.querySelector("#lab-observed-label").textContent = P_ALPHA_STATE.distribution === "binomial" ? copy.successes : copy.observed;
      root.querySelector("#lab-observed-output").textContent = P_ALPHA_STATE.distribution === "binomial" ? Math.round(P_ALPHA_STATE.observed) : Number(P_ALPHA_STATE.observed).toFixed(1);
      root.querySelector("#lab-alpha-output").textContent = P_ALPHA_STATE.alpha.toFixed(3); root.querySelector("#lab-alpha-card").textContent = P_ALPHA_STATE.alpha.toFixed(3); root.querySelector("#lab-p-output").textContent = formatProbability(p);
      const decision = root.querySelector("#lab-decision"); decision.textContent = rejected ? copy.reject : copy.retain; decision.className = rejected ? "is-reject" : "is-retain";
      root.querySelector("#lab-chart-container").innerHTML = config.discrete ? discreteNullChart(P_ALPHA_STATE.observed, P_ALPHA_STATE.alternative, P_ALPHA_STATE.alpha, copy) : continuousNullChart(P_ALPHA_STATE.distribution, P_ALPHA_STATE.observed, P_ALPHA_STATE.alternative, P_ALPHA_STATE.alpha, copy);
      const formula = root.querySelector("#lab-p-formula"); formula.dataset.tex = pAlphaFormula(P_ALPHA_STATE.distribution, P_ALPHA_STATE.alternative); formula.textContent = formula.dataset.tex;
      const note = P_ALPHA_STATE.distribution === "binomial"
        ? (LANG === "zh" ? `${copy.discreteMeaning} 这里的双侧精确 p 值按“零假设概率不大于观测结果”的结果求和，不能机械写成两倍单尾。` : `${copy.discreteMeaning} Here the two-sided exact p-value sums outcomes no more probable than the observed one; it is not mechanically twice one tail.`)
        : copy.pMeaning;
      root.querySelector("#lab-p-note").textContent = note; rerenderMath(root.querySelector(".p-alpha-lab")); if (reset) resetSimulation();
    };
    distribution.addEventListener("change", () => { P_ALPHA_STATE.distribution = distribution.value; const config = pAlphaConfig(distribution.value); P_ALPHA_STATE.observed = config.initial; if (distribution.value === "chiSquare") P_ALPHA_STATE.alternative = "greater"; else P_ALPHA_STATE.alternative = "two"; alternative.value = P_ALPHA_STATE.alternative; update(); });
    alternative.addEventListener("change", () => { P_ALPHA_STATE.alternative = alternative.value; update(); });
    observed.addEventListener("input", () => { P_ALPHA_STATE.observed = Number(observed.value); update(); });
    alpha.addEventListener("input", () => { P_ALPHA_STATE.alpha = Number(alpha.value); update(); });
    runButton.addEventListener("click", () => {
      resetSimulation(); const copy = P_ALPHA_TEXT[LANG]; P_ALPHA_STATE.run += 1; const random = seededRandom(20260717 + P_ALPHA_STATE.run * 7919); const results = Array.from({ length: 100 }, () => labPValue(P_ALPHA_STATE.distribution, sampleNull(P_ALPHA_STATE.distribution, random), P_ALPHA_STATE.alternative) <= P_ALPHA_STATE.alpha); const expected = theoreticalRejectionRate(P_ALPHA_STATE.distribution, P_ALPHA_STATE.alternative, P_ALPHA_STATE.alpha); const grid = root.querySelector("#lab-sim-grid"); const summary = root.querySelector("#lab-sim-summary");
      const draw = count => { const visible = results.slice(0, count); const falsePositives = visible.filter(Boolean).length; grid.innerHTML = visible.map(rejected => `<span class="sim-dot${rejected ? " is-reject" : ""}" aria-hidden="true">${rejected ? "×" : "•"}</span>`).join(""); summary.textContent = `${copy.falsePositives}: ${falsePositives}/${count}. ${copy.expected} ${(expected * 100).toFixed(1)}/100.`; };
      runButton.disabled = true; runButton.textContent = copy.running;
      if (matchMedia("(prefers-reduced-motion: reduce)").matches) { draw(100); runButton.disabled = false; runButton.textContent = copy.run; return; }
      let count = 0; P_ALPHA_STATE.timer = setInterval(() => { count = Math.min(100, count + 4); draw(count); if (count >= 100) { clearInterval(P_ALPHA_STATE.timer); P_ALPHA_STATE.timer = null; runButton.disabled = false; runButton.textContent = copy.run; } }, 32);
    });
    update(false);
  }
  function renderHelp() {
    const root = document.getElementById("app"); if (!root) return;
    document.title = LANG === "zh" ? "统计推断帮助中心｜假设检验指南" : "Statistical inference help | Hypothesis Testing Guide";
    root.innerHTML = `<section class="hero-block compact"><p class="eyebrow">${escapeHTML(UI_TEXT[LANG].navHelp)}</p><h1>${escapeHTML(UI_TEXT[LANG].helpTitle)}</h1><p class="lead">${escapeHTML(UI_TEXT[LANG].helpLead)}</p></section>${renderPAlphaStory()}${renderPAlphaLab()}<div class="help-grid">${HELP_CONTENT.filter(section => section.id !== "pvalue").map(section => `<section class="help-card" id="${section.id}"><h2>${escapeHTML(localized(section.title))}</h2>${listHtml(localized(section.body))}</section>`).join("")}</div>`;
    bindPAlphaStory(root); bindPAlphaLab(root); rerenderMath(root);
  }
  async function copyText(value, button) { try { await navigator.clipboard.writeText(value); const original = button.textContent; button.textContent = UI_TEXT[LANG].copied; setTimeout(() => { button.textContent = original; }, 1400); } catch (_) { status(UI_TEXT[LANG].copy); } }
  function boot() {
    syncLanguage(); document.getElementById("language-toggle")?.addEventListener("click", () => { LANG = LANG === "zh" ? "en" : "zh"; const url = new URL(location.href); url.searchParams.set("lang", LANG); history.replaceState({}, "", url); syncLanguage(); renderCurrent(); });
    document.addEventListener("keydown", event => {
      const searchShortcut = event.key === "/" || (event.key.toLowerCase() === "k" && (event.ctrlKey || event.metaKey));
      const editing = /input|textarea|select/i.test(document.activeElement?.tagName || "");
      if (searchShortcut && !editing) {
        event.preventDefault(); const search = document.querySelector("#method-search");
        if (search) search.focus(); else if (document.body.dataset.page === "index") location.href = safeUrl("index.html", { view: "library", focus: "search" });
      }
      if (event.key === "Escape" && document.activeElement?.id === "method-search") { document.activeElement.value = ""; document.activeElement.dispatchEvent(new Event("input")); }
    });
    window.addEventListener("popstate", renderCurrent); renderCurrent();
  }
  function renderCurrent() { syncLanguage(); const page = document.body.dataset.page; if (page === "method") renderMethod(); else if (page === "help") renderHelp(); else renderIndex(); const app = document.getElementById("app"); if (app) app.setAttribute("aria-busy", "false"); }
  boot();
})();

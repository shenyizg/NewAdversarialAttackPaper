# LLM / MLLM（语言模型） - 基础/可解释性
**update at 2026-01-25 10:36:50**

按分类器置信度从高到低排序。

## **1. AdversaRiskQA: An Adversarial Factuality Benchmark for High-Risk Domains**

AdversaRiskQA：面向高风险领域的对抗性事实性基准测试 cs.CL

13 pages, 4 figures, and 11 tables

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15511v1) [paper-pdf](https://arxiv.org/pdf/2601.15511v1)

**Confidence**: 0.95

**Authors**: Adam Szelestey, Sofie van Engelen, Tianhao Huang, Justin Snelders, Qintao Zeng, Songgaojun Deng

**Abstract**: Hallucination in large language models (LLMs) remains an acute concern, contributing to the spread of misinformation and diminished public trust, particularly in high-risk domains. Among hallucination types, factuality is crucial, as it concerns a model's alignment with established world knowledge. Adversarial factuality, defined as the deliberate insertion of misinformation into prompts with varying levels of expressed confidence, tests a model's ability to detect and resist confidently framed falsehoods. Existing work lacks high-quality, domain-specific resources for assessing model robustness under such adversarial conditions, and no prior research has examined the impact of injected misinformation on long-form text factuality.   To address this gap, we introduce AdversaRiskQA, the first verified and reliable benchmark systematically evaluating adversarial factuality across Health, Finance, and Law. The benchmark includes two difficulty levels to test LLMs' defensive capabilities across varying knowledge depths. We propose two automated methods for evaluating the adversarial attack success and long-form factuality. We evaluate six open- and closed-source LLMs from the Qwen, GPT-OSS, and GPT families, measuring misinformation detection rates. Long-form factuality is assessed on Qwen3 (30B) under both baseline and adversarial conditions. Results show that after excluding meaningless responses, Qwen3 (80B) achieves the highest average accuracy, while GPT-5 maintains consistently high accuracy. Performance scales non-linearly with model size, varies by domains, and gaps between difficulty levels narrow as models grow. Long-form evaluation reveals no significant correlation between injected misinformation and the model's factual output. AdversaRiskQA provides a valuable benchmark for pinpointing LLM weaknesses and developing more reliable models for high-stakes applications.

摘要: 大型语言模型（LLMs）中的幻觉问题依然严峻，尤其是在高风险领域，它助长了错误信息的传播并削弱了公众信任。在各类幻觉中，事实性至关重要，因为它关乎模型与既定世界知识的一致性。对抗性事实性定义为在提示中故意插入不同置信度表达的错误信息，以测试模型检测和抵抗自信表述虚假信息的能力。现有研究缺乏高质量、领域特定的资源来评估模型在此类对抗条件下的鲁棒性，且尚无研究考察注入错误信息对长文本事实性的影响。为填补这一空白，我们提出了AdversaRiskQA，这是首个经过验证且可靠的基准测试，系统评估了健康、金融和法律领域的对抗性事实性。该基准包含两个难度级别，以测试LLMs在不同知识深度下的防御能力。我们提出了两种自动化方法来评估对抗攻击成功率和长文本事实性。我们评估了来自Qwen、GPT-OSS和GPT家族的六个开源和闭源LLMs，测量了错误信息检测率。长文本事实性在Qwen3（30B）模型上进行了基线和对抗条件下的评估。结果显示，在排除无意义响应后，Qwen3（80B）实现了最高的平均准确率，而GPT-5保持了持续的高准确率。性能随模型规模呈非线性增长，因领域而异，且难度级别间的差距随模型增大而缩小。长文本评估显示，注入的错误信息与模型的事实性输出之间无显著相关性。AdversaRiskQA为精确定位LLMs弱点及开发更可靠的高风险应用模型提供了有价值的基准。



## **2. Lightweight LLMs for Network Attack Detection in IoT Networks**

面向物联网网络攻击检测的轻量化大语言模型 cs.CR

6 pages with 2 figures, This paper was accepted and presented at the 7th Computing, Communications and IoT Applications Conference (ComComAp 2025), held in Madrid, Spain, during 14th to 17th December 2025

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15269v1) [paper-pdf](https://arxiv.org/pdf/2601.15269v1)

**Confidence**: 0.95

**Authors**: Piyumi Bhagya Sudasinghe, Kushan Sudheera Kalupahana Liyanage, Harsha S. Gardiyawasam Pussewalage

**Abstract**: The rapid growth of Internet of Things (IoT) devices has increased the scale and diversity of cyberattacks, exposing limitations in traditional intrusion detection systems. Classical machine learning (ML) models such as Random Forest and Support Vector Machine perform well on known attacks but require retraining to detect unseen or zero-day threats. This study investigates lightweight decoder-only Large Language Models (LLMs) for IoT attack detection by integrating structured-to-text conversion, Quantized Low-Rank Adaptation (QLoRA) fine-tuning, and Retrieval-Augmented Generation (RAG). Network traffic features are transformed into compact natural-language prompts, enabling efficient adaptation under constrained hardware. Experiments on the CICIoT2023 dataset show that a QLoRA-tuned LLaMA-1B model achieves an F1-score of 0.7124, comparable to the Random Forest (RF) baseline (0.7159) for known attacks. With RAG, the system attains 42.63% accuracy on unseen attack types without additional training, demonstrating practical zero-shot capability. These results highlight the potential of retrieval-enhanced lightweight LLMs as adaptable and resource-efficient solutions for next-generation IoT intrusion detection.

摘要: 物联网设备的快速增长扩大了网络攻击的规模和多样性，暴露出传统入侵检测系统的局限性。随机森林和支持向量机等经典机器学习模型在已知攻击检测上表现良好，但需要重新训练才能检测未知或零日威胁。本研究通过集成结构化到文本转换、量化低秩适配微调和检索增强生成技术，探索轻量化仅解码器大语言模型在物联网攻击检测中的应用。网络流量特征被转换为紧凑的自然语言提示，实现在受限硬件下的高效适配。在CICIoT2023数据集上的实验表明，QLoRA微调的LLaMA-1B模型在已知攻击检测中达到0.7124的F1分数，与随机森林基线（0.7159）相当。结合检索增强生成技术，该系统在无需额外训练的情况下对未知攻击类型达到42.63%的检测准确率，展现出实用的零样本能力。这些结果凸显了检索增强型轻量化大语言模型作为下一代物联网入侵检测的适应性强、资源高效解决方案的潜力。



## **3. Turn-Based Structural Triggers: Prompt-Free Backdoors in Multi-Turn LLMs**

基于回合的结构化触发器：多轮LLM中的无提示后门攻击 cs.CR

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.14340v1) [paper-pdf](https://arxiv.org/pdf/2601.14340v1)

**Confidence**: 0.95

**Authors**: Yiyang Lu, Jinwen He, Yue Zhao, Kai Chen, Ruigang Liang

**Abstract**: Large Language Models (LLMs) are widely integrated into interactive systems such as dialogue agents and task-oriented assistants. This growing ecosystem also raises supply-chain risks, where adversaries can distribute poisoned models that degrade downstream reliability and user trust. Existing backdoor attacks and defenses are largely prompt-centric, focusing on user-visible triggers while overlooking structural signals in multi-turn conversations. We propose Turn-based Structural Trigger (TST), a backdoor attack that activates from dialogue structure, using the turn index as the trigger and remaining independent of user inputs. Across four widely used open-source LLM models, TST achieves an average attack success rate (ASR) of 99.52% with minimal utility degradation, and remains effective under five representative defenses with an average ASR of 98.04%. The attack also generalizes well across instruction datasets, maintaining an average ASR of 99.19%. Our results suggest that dialogue structure constitutes an important and under-studied attack surface for multi-turn LLM systems, motivating structure-aware auditing and mitigation in practice.

摘要: 大型语言模型（LLMs）已广泛应用于对话代理和任务导向助手等交互式系统。这一不断发展的生态系统也带来了供应链风险，攻击者可能分发被投毒模型，从而损害下游可靠性和用户信任。现有的后门攻击与防御主要围绕提示展开，关注用户可见的触发器，却忽视了多轮对话中的结构化信号。我们提出基于回合的结构化触发器（TST），这是一种利用对话结构激活的后门攻击，以回合索引作为触发器，且独立于用户输入。在四种广泛使用的开源LLM模型上，TST实现了平均99.52%的攻击成功率（ASR），且性能损失极小；在五种代表性防御下仍保持平均98.04%的ASR。该攻击在不同指令数据集上也表现出良好的泛化能力，平均ASR达99.19%。我们的研究表明，对话结构构成了多轮LLM系统中一个重要且尚未被充分研究的攻击面，这为实践中开展结构感知的审计与缓解措施提供了依据。



## **4. OI-Bench: An Option Injection Benchmark for Evaluating LLM Susceptibility to Directive Interference**

OI-Bench：用于评估大语言模型对指令干扰敏感性的选项注入基准 cs.CL

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13300v1) [paper-pdf](https://arxiv.org/pdf/2601.13300v1)

**Confidence**: 0.95

**Authors**: Yow-Fu Liou, Yu-Chien Tang, Yu-Hsiang Liu, An-Zi Yen

**Abstract**: Benchmarking large language models (LLMs) is critical for understanding their capabilities, limitations, and robustness. In addition to interface artifacts, prior studies have shown that LLM decisions can be influenced by directive signals such as social cues, framing, and instructions. In this work, we introduce option injection, a benchmarking approach that augments the multiple-choice question answering (MCQA) interface with an additional option containing a misleading directive, leveraging standardized choice structure and scalable evaluation. We construct OI-Bench, a benchmark of 3,000 questions spanning knowledge, reasoning, and commonsense tasks, with 16 directive types covering social compliance, bonus framing, threat framing, and instructional interference. This setting combines manipulation of the choice interface with directive-based interference, enabling systematic assessment of model susceptibility. We evaluate 12 LLMs to analyze attack success rates, behavioral responses, and further investigate mitigation strategies ranging from inference-time prompting to post-training alignment. Experimental results reveal substantial vulnerabilities and heterogeneous robustness across models. OI-Bench is expected to support more systematic evaluation of LLM robustness to directive interference within choice-based interfaces.

摘要: 对大语言模型（LLMs）进行基准测试对于理解其能力、局限性和鲁棒性至关重要。除了界面伪影外，先前研究表明LLM的决策可能受到社交线索、框架效应和指令等导向信号的影响。本研究提出选项注入方法——一种通过在多选题（MCQA）界面中增加包含误导性指令的额外选项，利用标准化选择结构和可扩展评估的基准测试方法。我们构建了OI-Bench基准，包含涵盖知识、推理和常识任务的3,000个问题，涉及社交依从性、奖励框架、威胁框架和指令干扰等16种指令类型。该设置将选择界面操纵与基于指令的干扰相结合，能够系统评估模型的敏感性。我们评估了12个LLM以分析攻击成功率、行为响应，并进一步研究从推理时提示到训练后对齐的缓解策略。实验结果显示模型存在显著脆弱性且鲁棒性存在异质性。OI-Bench有望支持对基于选择的界面中LLM抗指令干扰鲁棒性进行更系统化的评估。



## **5. Adversarial News and Lost Profits: Manipulating Headlines in LLM-Driven Algorithmic Trading**

对抗性新闻与利润损失：在LLM驱动的算法交易中操纵新闻标题 cs.CR

This work has been accepted for publication at the IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). The final version will be available on IEEE Xplore

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13082v1) [paper-pdf](https://arxiv.org/pdf/2601.13082v1)

**Confidence**: 0.95

**Authors**: Advije Rizvani, Giovanni Apruzzese, Pavel Laskov

**Abstract**: Large Language Models (LLMs) are increasingly adopted in the financial domain. Their exceptional capabilities to analyse textual data make them well-suited for inferring the sentiment of finance-related news. Such feedback can be leveraged by algorithmic trading systems (ATS) to guide buy/sell decisions. However, this practice bears the risk that a threat actor may craft "adversarial news" intended to mislead an LLM. In particular, the news headline may include "malicious" content that remains invisible to human readers but which is still ingested by the LLM. Although prior work has studied textual adversarial examples, their system-wide impact on LLM-supported ATS has not yet been quantified in terms of monetary risk. To address this threat, we consider an adversary with no direct access to an ATS but able to alter stock-related news headlines on a single day. We evaluate two human-imperceptible manipulations in a financial context: Unicode homoglyph substitutions that misroute models during stock-name recognition, and hidden-text clauses that alter the sentiment of the news headline. We implement a realistic ATS in Backtrader that fuses an LSTM-based price forecast with LLM-derived sentiment (FinBERT, FinGPT, FinLLaMA, and six general-purpose LLMs), and quantify monetary impact using portfolio metrics. Experiments on real-world data show that manipulating a one-day attack over 14 months can reliably mislead LLMs and reduce annual returns by up to 17.7 percentage points. To assess real-world feasibility, we analyze popular scraping libraries and trading platforms and survey 27 FinTech practitioners, confirming our hypotheses. We notified trading platform owners of this security issue.

摘要: 大型语言模型（LLMs）在金融领域的应用日益广泛。其分析文本数据的卓越能力使其非常适合推断财经新闻的情感倾向。算法交易系统（ATS）可利用此类反馈指导买卖决策。然而，这种做法存在风险，即威胁行为者可能制作旨在误导LLM的“对抗性新闻”。具体而言，新闻标题可能包含对人类读者不可见但仍被LLM摄入的“恶意”内容。尽管先前研究已探讨文本对抗样本，但其对LLM支持的ATS的系统性影响尚未从货币风险角度量化。为应对此威胁，我们考虑一个无法直接访问ATS但能在单日内篡改股票相关新闻标题的对手。我们在金融背景下评估两种人类难以察觉的操纵手段：在股票名称识别过程中误导模型的Unicode同形异义词替换，以及改变新闻标题情感倾向的隐藏文本条款。我们在Backtrader中实现了一个现实的ATS，融合基于LSTM的价格预测与LLM衍生的情感分析（使用FinBERT、FinGPT、FinLLaMA及六个通用LLM），并通过投资组合指标量化货币影响。基于真实数据的实验表明，在14个月内操纵单日攻击可可靠地误导LLMs，并使年化收益率降低高达17.7个百分点。为评估现实可行性，我们分析了主流爬虫库和交易平台，并调研了27位金融科技从业者，结果验证了我们的假设。我们已就此安全问题通知交易平台所有者。



## **6. On the Evidentiary Limits of Membership Inference for Copyright Auditing**

论成员推理在版权审计中的证据局限性 cs.CR

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.12937v1) [paper-pdf](https://arxiv.org/pdf/2601.12937v1)

**Confidence**: 0.95

**Authors**: Murat Bilgehan Ertan, Emirhan Böge, Min Chen, Kaleel Mahmood, Marten van Dijk

**Abstract**: As large language models (LLMs) are trained on increasingly opaque corpora, membership inference attacks (MIAs) have been proposed to audit whether copyrighted texts were used during training, despite growing concerns about their reliability under realistic conditions. We ask whether MIAs can serve as admissible evidence in adversarial copyright disputes where an accused model developer may obfuscate training data while preserving semantic content, and formalize this setting through a judge-prosecutor-accused communication protocol. To test robustness under this protocol, we introduce SAGE (Structure-Aware SAE-Guided Extraction), a paraphrasing framework guided by Sparse Autoencoders (SAEs) that rewrites training data to alter lexical structure while preserving semantic content and downstream utility. Our experiments show that state-of-the-art MIAs degrade when models are fine-tuned on SAGE-generated paraphrases, indicating that their signals are not robust to semantics-preserving transformations. While some leakage remains in certain fine-tuning regimes, these results suggest that MIAs are brittle in adversarial settings and insufficient, on their own, as a standalone mechanism for copyright auditing of LLMs.

摘要: 随着大语言模型（LLMs）在日益不透明的语料库上进行训练，尽管在实际条件下其可靠性备受质疑，成员推理攻击（MIAs）仍被提议用于审计训练过程中是否使用了受版权保护的文本。我们探讨在对抗性版权纠纷中，当被指控的模型开发者可能对训练数据进行语义保留的模糊处理时，MIAs能否作为可采纳的证据，并通过法官-检察官-被告的通信协议形式化这一场景。为测试该协议下的鲁棒性，我们提出了SAGE（结构感知的稀疏自编码器引导提取）——一种由稀疏自编码器（SAEs）引导的复述框架，该框架通过改写训练数据来改变词汇结构，同时保留语义内容和下游效用。实验表明，当模型在SAGE生成的复述文本上进行微调时，最先进的MIAs性能显著下降，表明其信号对语义保留的转换不具备鲁棒性。尽管在某些微调机制中仍存在部分信息泄露，但这些结果表明MIAs在对抗性环境中具有脆弱性，无法作为LLMs版权审计的独立机制。



## **7. Less Is More -- Until It Breaks: Security Pitfalls of Vision Token Compression in Large Vision-Language Models**

少即是多——直至崩溃：大型视觉语言模型中视觉令牌压缩的安全隐患 cs.CR

**SubmitDate**: 2026-01-17    [abs](http://arxiv.org/abs/2601.12042v1) [paper-pdf](https://arxiv.org/pdf/2601.12042v1)

**Confidence**: 0.95

**Authors**: Xiaomei Zhang, Zhaoxi Zhang, Leo Yu Zhang, Yanjun Zhang, Guanhong Tao, Shirui Pan

**Abstract**: Visual token compression is widely adopted to improve the inference efficiency of Large Vision-Language Models (LVLMs), enabling their deployment in latency-sensitive and resource-constrained scenarios. However, existing work has mainly focused on efficiency and performance, while the security implications of visual token compression remain largely unexplored. In this work, we first reveal that visual token compression substantially degrades the robustness of LVLMs: models that are robust under uncompressed inference become highly vulnerable once compression is enabled. These vulnerabilities are state-specific; failure modes emerge only in the compressed setting and completely disappear when compression is disabled, making them particularly hidden and difficult to diagnose. By analyzing the key stages of the compression process, we identify instability in token importance ranking as the primary cause of this robustness degradation. Small and imperceptible perturbations can significantly alter token rankings, leading the compression mechanism to mistakenly discard task-critical information and ultimately causing model failure. Motivated by this observation, we propose a Compression-Aware Attack to systematically study and exploit this vulnerability. CAA directly targets the token selection mechanism and induces failures exclusively under compressed inference. We further extend this approach to more realistic black-box settings and introduce Transfer CAA, where neither the target model nor the compression configuration is accessible. We further evaluate potential defenses and find that they provide only limited protection. Extensive experiments across models, datasets, and compression methods show that visual token compression significantly undermines robustness, revealing a previously overlooked efficiency-security trade-off.

摘要: 视觉令牌压缩技术被广泛用于提升大型视觉语言模型（LVLMs）的推理效率，使其能够部署在延迟敏感和资源受限的场景中。然而，现有研究主要关注效率和性能，而视觉令牌压缩的安全影响在很大程度上尚未得到探索。在本研究中，我们首次揭示视觉令牌压缩会显著降低LVLMs的鲁棒性：在未压缩推理下表现稳健的模型，一旦启用压缩就会变得高度脆弱。这些漏洞具有状态特异性：失效模式仅在压缩设置下出现，并在禁用压缩时完全消失，这使得它们特别隐蔽且难以诊断。通过分析压缩过程的关键阶段，我们发现令牌重要性排序的不稳定性是导致鲁棒性下降的主要原因。微小且难以察觉的扰动可以显著改变令牌排序，导致压缩机制错误地丢弃任务关键信息，最终引发模型失效。基于这一观察，我们提出了一种压缩感知攻击（CAA）来系统研究和利用此漏洞。CAA直接针对令牌选择机制，并仅在压缩推理下诱发失效。我们进一步将此方法扩展到更现实的的黑盒设置中，提出了迁移CAA，其中既无法访问目标模型也无法获取压缩配置。我们还评估了潜在的防御措施，发现它们仅能提供有限的保护。跨模型、数据集和压缩方法的广泛实验表明，视觉令牌压缩显著削弱了鲁棒性，揭示了一个先前被忽视的效率-安全权衡问题。



## **8. Membership Inference on LLMs in the Wild**

实际场景中大型语言模型的成员推断攻击 cs.CL

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11314v1) [paper-pdf](https://arxiv.org/pdf/2601.11314v1)

**Confidence**: 0.95

**Authors**: Jiatong Yi, Yanyang Li

**Abstract**: Membership Inference Attacks (MIAs) act as a crucial auditing tool for the opaque training data of Large Language Models (LLMs). However, existing techniques predominantly rely on inaccessible model internals (e.g., logits) or suffer from poor generalization across domains in strict black-box settings where only generated text is available. In this work, we propose SimMIA, a robust MIA framework tailored for this text-only regime by leveraging an advanced sampling strategy and scoring mechanism. Furthermore, we present WikiMIA-25, a new benchmark curated to evaluate MIA performance on modern proprietary LLMs. Experiments demonstrate that SimMIA achieves state-of-the-art results in the black-box setting, rivaling baselines that exploit internal model information.

摘要: 成员推断攻击（MIAs）作为大型语言模型（LLMs）不透明训练数据的关键审计工具，现有技术主要依赖于难以获取的模型内部信息（如logits），或在仅能获取生成文本的严格黑盒设置中面临跨领域泛化能力不足的问题。本研究提出SimMIA，一种通过先进采样策略和评分机制专门针对纯文本场景设计的鲁棒MIA框架。此外，我们构建了WikiMIA-25新基准，用于评估现代专有LLMs上的MIA性能。实验表明，SimMIA在黑盒设置中取得了最先进的结果，其性能可与利用模型内部信息的基线方法相媲美。



## **9. CoSPED: Consistent Soft Prompt Targeted Data Extraction and Defense**

CoSPED：一致性软提示目标数据提取与防御 cs.CR

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2510.11137v3) [paper-pdf](https://arxiv.org/pdf/2510.11137v3)

**Confidence**: 0.95

**Authors**: Zhuochen Yang, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Large language models have gained widespread attention recently, but their potential security vulnerabilities, especially privacy leakage, are also becoming apparent. To test and evaluate for data extraction risks in LLM, we proposed CoSPED, short for Consistent Soft Prompt targeted data Extraction and Defense. We introduce several innovative components, including Dynamic Loss, Additive Loss, Common Loss, and Self Consistency Decoding Strategy, and tested to enhance the consistency of the soft prompt tuning process. Through extensive experimentation with various combinations, we achieved an extraction rate of 65.2% at a 50-token prefix comparison. Our comparisons of CoSPED with other reference works confirm our superior extraction rates. We evaluate CoSPED on more scenarios, achieving Pythia model extraction rate of 51.7% and introducing cross-model comparison. Finally, we explore defense through Rank-One Model Editing and achieve a reduction in the extraction rate to 1.6%, which proves that our analysis of extraction mechanisms can directly inform effective mitigation strategies against soft prompt-based attacks.

摘要: 大语言模型近期获得广泛关注，但其潜在安全漏洞，尤其是隐私泄露问题也日益凸显。为测试和评估LLM中的数据提取风险，我们提出了CoSPED（一致性软提示目标数据提取与防御）。我们引入了动态损失、加性损失、公共损失及自一致性解码策略等多个创新组件，通过测试增强了软提示调优过程的一致性。通过多种组合的广泛实验，我们在50个token前缀比较中实现了65.2%的提取率。CoSPED与其他参考工作的对比证实了我们更高的提取率。我们在更多场景中评估CoSPED，实现了Pythia模型51.7%的提取率，并引入了跨模型比较。最后，我们通过Rank-One模型编辑探索防御方案，将提取率降低至1.6%，这证明我们对提取机制的分析可直接为基于软提示攻击的有效缓解策略提供依据。



## **10. Membership Inference Attacks on LLM-based Recommender Systems**

基于大语言模型的推荐系统中的成员推理攻击 cs.IR

This is paper is under review ACL 2026

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2508.18665v5) [paper-pdf](https://arxiv.org/pdf/2508.18665v5)

**Confidence**: 0.95

**Authors**: Jiajie He, Min-Chun Chen, Xintong Chen, Xinyang Fang, Yuechun Gu, Keke Chen

**Abstract**: Large language models (LLMs) based recommender systems (RecSys) can adapt to different domains flexibly. It utilizes in-context learning (ICL), i.e., prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, encompassing implicit feedback such as clicked items and explicit product reviews. Such private information may be exposed by novel privacy attacks. However, no study has been conducted on this important issue. We design several membership inference attacks (MIAs) aimed to revealing whether system prompts include victims' historical interactions. The attacks are \emph{Similarity, Memorization, Inquiry, and Poisoning attacks}, each utilizing unique features of LLMs or RecSys. We have carefully evaluated them on five of the latest open-source LLMs and three well-known RecSys benchmark datasets. The results confirm that the MIA threat to LLM RecSys is realistic: inquiry and poisoning attacks show significantly high attack advantages. We also discussed possible methods to mitigate such MIA threats. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts, the position of the victim in the shots, the number of poisoning items in the prompt,etc.

摘要: 基于大语言模型（LLMs）的推荐系统（RecSys）能够灵活适应不同领域。该系统利用上下文学习（ICL），即提示词，来定制推荐功能，其中包含敏感的历史用户特定物品交互信息，涵盖点击物品等隐式反馈和产品评论等显式反馈。此类隐私信息可能遭受新型隐私攻击的泄露。然而，目前尚未有研究针对这一重要问题展开探讨。我们设计了多种成员推理攻击（MIAs），旨在揭示系统提示词是否包含受害者的历史交互记录。这些攻击包括相似性攻击、记忆攻击、查询攻击和投毒攻击，每种攻击均利用了LLMs或RecSys的独特特性。我们在五种最新的开源LLMs和三个知名RecSys基准数据集上进行了细致评估。结果证实，针对LLM RecSys的MIA威胁是真实存在的：查询攻击和投毒攻击显示出显著的高攻击优势。我们还探讨了可能的缓解此类MIA威胁的方法，并分析了影响这些攻击的因素，例如系统提示词中的示例数量、受害者在示例中的位置、提示词中投毒物品的数量等。



## **11. EVADE-Bench: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

EVADE-Bench：面向电子商务应用中规避内容检测的多模态基准 cs.CL

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2505.17654v3) [paper-pdf](https://arxiv.org/pdf/2505.17654v3)

**Confidence**: 0.95

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyu Chang, Hamid Alinejad-Rokny, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.

摘要: 电子商务平台日益依赖大型语言模型（LLMs）和视觉语言模型（VLMs）来检测非法或误导性产品内容。然而，这些模型在面对规避内容时仍然脆弱：这类输入（文本或图像）表面上符合平台政策，却暗中传达被禁止的主张。与引发明显失败的传统对抗性攻击不同，规避内容利用模糊性和上下文，使其检测难度大大增加。现有的鲁棒性基准对这一高要求的现实挑战指导有限。我们推出了EVADE，这是首个由专家策划、面向中文、多模态的基准，专门用于评估基础模型在电子商务中规避内容检测的能力。该数据集包含2,833个标注文本样本和13,961张图像，涵盖六个高要求的产品类别，包括塑身、增高和健康补充剂。两个互补的任务评估了不同的能力：单违规任务，通过简短提示探究细粒度推理；以及一体化任务，通过将重叠的政策规则合并为统一指令来测试长上下文推理。值得注意的是，一体化设置显著缩小了部分匹配与完全匹配准确率之间的性能差距，表明更清晰的规则定义能改善人类与模型判断之间的一致性。我们对26个主流LLMs和VLMs进行了基准测试，观察到显著的性能差距：即使是最先进的模型也经常错误分类规避样本。通过发布EVADE和强基线，我们为评估规避内容检测提供了首个严格标准，揭示了当前多模态推理的根本局限性，并为电子商务中更安全、更透明的内容审核系统奠定了基础。数据集公开于 https://huggingface.co/datasets/koenshen/EVADE-Bench。



## **12. FDLLM: A Dedicated Detector for Black-Box LLMs Fingerprinting**

FDLLM：一种专用于黑盒大语言模型指纹识别的检测器 cs.CR

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2501.16029v4) [paper-pdf](https://arxiv.org/pdf/2501.16029v4)

**Confidence**: 0.95

**Authors**: Zhiyuan Fu, Junfan Chen, Lan Zhang, Ting Yang, Jun Niu, Hongyu Sun, Ruidong Li, Peng Liu, Jice Wang, Fannv He, Qiuling Yue, Yuqing Zhang

**Abstract**: Large Language Models (LLMs) are rapidly transforming the landscape of digital content creation. However, the prevalent black-box Application Programming Interface (API) access to many LLMs introduces significant challenges in accountability, governance, and security. LLM fingerprinting, which aims to identify the source model by analyzing statistical and stylistic features of generated text, offers a potential solution. Current progress in this area is hindered by a lack of dedicated datasets and the need for efficient, practical methods that are robust against adversarial manipulations. To address these challenges, we introduce FD-Dataset, a comprehensive bilingual fingerprinting benchmark comprising 90,000 text samples from 20 famous proprietary and open-source LLMs. Furthermore, we present FDLLM, a novel fingerprinting method that leverages parameter-efficient Low-Rank Adaptation (LoRA) to fine-tune a foundation model. This approach enables LoRA to extract deep, persistent features that characterize each source LLM. Through our analysis, we find that LoRA adaptation promotes the aggregation of outputs from the same LLM in representation space while enhancing the separation between different LLMs. This mechanism explains why LoRA proves particularly effective for LLM fingerprinting. Extensive empirical evaluations on FD-Dataset demonstrate FDLLM's superiority, achieving a Macro F1 score 22.1% higher than the strongest baseline. FDLLM also exhibits strong generalization to newly released models, achieving an average accuracy of 95% on unseen models. Notably, FDLLM remains consistently robust under various adversarial attacks, including polishing, translation, and synonym substitution. Experimental results show that FDLLM reduces the average attack success rate from 49.2% (LM-D) to 23.9%.

摘要: 大语言模型（LLMs）正在迅速改变数字内容创作的格局。然而，许多LLMs普遍采用的黑盒应用程序编程接口（API）访问方式，给问责、治理和安全带来了重大挑战。LLM指纹识别通过分析生成文本的统计和风格特征来识别源模型，为这一问题提供了潜在解决方案。当前该领域进展受到专用数据集缺乏以及需要高效、实用且能抵抗对抗性操作的方法所阻碍。为解决这些挑战，我们引入了FD-Dataset，这是一个全面的双语指纹识别基准数据集，包含来自20个知名专有和开源LLMs的90,000个文本样本。此外，我们提出了FDLLM，一种新颖的指纹识别方法，利用参数高效的LoRA（低秩适应）技术对基础模型进行微调。该方法使LoRA能够提取表征每个源LLM的深层持久特征。通过分析，我们发现LoRA适应促进了同一LLM输出在表示空间中的聚合，同时增强了不同LLM之间的分离性。这一机制解释了为何LoRA在LLM指纹识别中特别有效。在FD-Dataset上进行的大量实证评估表明，FDLLM具有显著优势，其宏平均F1分数比最强基线高出22.1%。FDLLM对新发布模型也表现出强大的泛化能力，在未见模型上平均准确率达到95%。值得注意的是，FDLLM在各种对抗攻击（包括润色、翻译和同义词替换）下始终保持稳健性。实验结果显示，FDLLM将平均攻击成功率从49.2%（LM-D）降低至23.9%。



## **13. Undesirable Memorization in Large Language Models: A Survey**

大型语言模型中的不良记忆现象：综述 cs.CL

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2410.02650v3) [paper-pdf](https://arxiv.org/pdf/2410.02650v3)

**Confidence**: 0.95

**Authors**: Ali Satvaty, Suzan Verberne, Fatih Turkmen

**Abstract**: While recent research increasingly showcases the remarkable capabilities of Large Language Models (LLMs), it is equally crucial to examine their associated risks. Among these, privacy and security vulnerabilities are particularly concerning, posing significant ethical and legal challenges. At the heart of these vulnerabilities stands memorization, which refers to a model's tendency to store and reproduce phrases from its training data. This phenomenon has been shown to be a fundamental source to various privacy and security attacks against LLMs. In this paper, we provide a taxonomy of the literature on LLM memorization, exploring it across three dimensions: granularity, retrievability, and desirability. Next, we discuss the metrics and methods used to quantify memorization, followed by an analysis of the causes and factors that contribute to memorization phenomenon. We then explore strategies that are used so far to mitigate the undesirable aspects of this phenomenon. We conclude our survey by identifying potential research topics for the near future, including methods to balance privacy and performance, and the analysis of memorization in specific LLM contexts such as conversational agents, retrieval-augmented generation, and diffusion language models. Given the rapid research pace in this field, we also maintain a dedicated repository of the references discussed in this survey which will be regularly updated to reflect the latest developments.

摘要: 尽管近期研究日益展示出大型语言模型（LLMs）的卓越能力，但审视其相关风险同样至关重要。其中，隐私与安全漏洞尤为令人担忧，构成了重大的伦理与法律挑战。这些漏洞的核心在于记忆现象，即模型倾向于存储并复现其训练数据中的语段。研究表明，这一现象是引发针对LLMs的各种隐私与安全攻击的根本来源。本文对LLM记忆现象的相关文献进行了分类梳理，从三个维度展开探讨：粒度、可检索性与合意性。随后，我们讨论了用于量化记忆现象的指标与方法，并分析了导致该现象的原因与影响因素。接着，我们探讨了目前用于缓解该现象不良影响的策略。最后，我们指出了近期潜在的研究方向，包括平衡隐私与性能的方法，以及在特定LLM场景（如对话代理、检索增强生成和扩散语言模型）中的记忆现象分析。鉴于该领域研究进展迅速，我们还维护了本综述所涉参考文献的专用存储库，并将定期更新以反映最新进展。



## **14. BenchOverflow: Measuring Overflow in Large Language Models via Plain-Text Prompts**

BenchOverflow：通过纯文本提示测量大语言模型的输出溢出 cs.CL

Accepted at TMLR 2026

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08490v1) [paper-pdf](https://arxiv.org/pdf/2601.08490v1)

**Confidence**: 0.95

**Authors**: Erin Feiglin, Nir Hutnik, Raz Lapid

**Abstract**: We investigate a failure mode of large language models (LLMs) in which plain-text prompts elicit excessive outputs, a phenomenon we term Overflow. Unlike jailbreaks or prompt injection, Overflow arises under ordinary interaction settings and can lead to elevated serving cost, latency, and cross-user performance degradation, particularly when scaled across many requests. Beyond usability, the stakes are economic and environmental: unnecessary tokens increase per-request cost and energy consumption, compounding into substantial operational spend and carbon footprint at scale. Moreover, Overflow represents a practical vector for compute amplification and service degradation in shared environments. We introduce BenchOverflow, a model-agnostic benchmark of nine plain-text prompting strategies that amplify output volume without adversarial suffixes or policy circumvention. Using a standardized protocol with a fixed budget of 5000 new tokens, we evaluate nine open- and closed-source models and observe pronounced rightward shifts and heavy tails in length distributions. Cap-saturation rates (CSR@1k/3k/5k) and empirical cumulative distribution functions (ECDFs) quantify tail risk; within-prompt variance and cross-model correlations show that Overflow is broadly reproducible yet heterogeneous across families and attack vectors. A lightweight mitigation-a fixed conciseness reminder-attenuates right tails and lowers CSR for all strategies across the majority of models. Our findings position length control as a measurable reliability, cost, and sustainability concern rather than a stylistic quirk. By enabling standardized comparison of length-control robustness across models, BenchOverflow provides a practical basis for selecting deployments that minimize resource waste and operating expense, and for evaluating defenses that curb compute amplification without eroding task performance.

摘要: 我们研究了大语言模型（LLMs）的一种故障模式，即纯文本提示会引发过量输出，我们将此现象称为“溢出”。与越狱或提示注入不同，溢出在普通交互设置下就会出现，可能导致服务成本增加、延迟升高以及跨用户性能下降，特别是在大规模请求时。除了可用性问题，其影响还涉及经济和环境层面：不必要的标记会增加单次请求的成本和能耗，在大规模部署时会累积成可观的运营支出和碳足迹。此外，溢出在共享环境中是计算放大和服务降级的实际载体。我们提出了BenchOverflow，这是一个模型无关的基准测试，包含九种纯文本提示策略，无需对抗性后缀或策略规避即可放大输出量。通过采用固定5000个新标记的标准化协议，我们评估了九个开源和闭源模型，观察到长度分布出现明显的右移和重尾现象。容量饱和率（CSR@1k/3k/5k）和经验累积分布函数（ECDF）量化了尾部风险；提示内方差和跨模型相关性表明，溢出现象在模型家族和攻击向量之间具有广泛可重现性但存在异质性。一种轻量级缓解措施——固定的简洁性提醒——能够减弱右尾并降低大多数模型中所有策略的CSR。我们的研究将长度控制定位为可测量的可靠性、成本和可持续性问题，而非风格偏好。通过实现跨模型长度控制鲁棒性的标准化比较，BenchOverflow为选择最小化资源浪费和运营成本的部署方案，以及评估在不削弱任务性能的前提下抑制计算放大的防御措施提供了实用基础。



## **15. A Semantic Decoupling-Based Two-Stage Rainy-Day Attack for Revealing Weather Robustness Deficiencies in Vision-Language Models**

基于语义解耦的两阶段雨天攻击：揭示视觉语言模型在天气鲁棒性方面的缺陷 cs.CV

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13238v1) [paper-pdf](https://arxiv.org/pdf/2601.13238v1)

**Confidence**: 0.95

**Authors**: Chengyin Hu, Xiang Chen, Zhe Jia, Weiwen Shi, Fengyu Zhang, Jiujiang Guo, Yiwei Wei

**Abstract**: Vision-Language Models (VLMs) are trained on image-text pairs collected under canonical visual conditions and achieve strong performance on multimodal tasks. However, their robustness to real-world weather conditions, and the stability of cross-modal semantic alignment under such structured perturbations, remain insufficiently studied. In this paper, we focus on rainy scenarios and introduce the first adversarial framework that exploits realistic weather to attack VLMs, using a two-stage, parameterized perturbation model based on semantic decoupling to analyze rain-induced shifts in decision-making. In Stage 1, we model the global effects of rainfall by applying a low-dimensional global modulation to condition the embedding space and gradually weaken the original semantic decision boundaries. In Stage 2, we introduce structured rain variations by explicitly modeling multi-scale raindrop appearance and rainfall-induced illumination changes, and optimize the resulting non-differentiable weather space to induce stable semantic shifts. Operating in a non-pixel parameter space, our framework generates perturbations that are both physically grounded and interpretable. Experiments across multiple tasks show that even physically plausible, highly constrained weather perturbations can induce substantial semantic misalignment in mainstream VLMs, posing potential safety and reliability risks in real-world deployment. Ablations further confirm that illumination modeling and multi-scale raindrop structures are key drivers of these semantic shifts.

摘要: 视觉语言模型（VLMs）在标准视觉条件下收集的图像-文本对上进行训练，在多模态任务中表现出色。然而，它们对真实世界天气条件的鲁棒性，以及在这种结构化扰动下跨模态语义对齐的稳定性，仍未得到充分研究。本文聚焦雨天场景，首次提出利用真实天气条件攻击VLMs的对抗性框架。该框架采用基于语义解耦的两阶段参数化扰动模型，分析降雨引起的决策偏移。第一阶段，通过低维全局调制建模降雨的全局效应，以调节嵌入空间并逐步削弱原始语义决策边界。第二阶段，通过显式建模多尺度雨滴外观和降雨引起的照明变化，引入结构化降雨变异，并优化由此产生的不可微天气空间以诱导稳定的语义偏移。我们的框架在非像素参数空间中运行，生成的扰动既具有物理基础又可解释。跨多个任务的实验表明，即使是物理上合理、高度受限的天气扰动，也能在主流VLMs中引发显著的语义错位，在实际部署中构成潜在的安全性和可靠性风险。消融实验进一步证实，照明建模和多尺度雨滴结构是这些语义偏移的关键驱动因素。



## **16. Hierarchical Refinement of Universal Multimodal Attacks on Vision-Language Models**

视觉语言模型通用多模态攻击的分层优化方法 cs.CV

15 pages, 7 figures

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10313v1) [paper-pdf](https://arxiv.org/pdf/2601.10313v1)

**Confidence**: 0.95

**Authors**: Peng-Fei Zhang, Zi Huang

**Abstract**: Existing adversarial attacks for VLP models are mostly sample-specific, resulting in substantial computational overhead when scaled to large datasets or new scenarios. To overcome this limitation, we propose Hierarchical Refinement Attack (HRA), a multimodal universal attack framework for VLP models. HRA refines universal adversarial perturbations (UAPs) at both the sample level and the optimization level. For the image modality, we disentangle adversarial examples into clean images and perturbations, allowing each component to be handled independently for more effective disruption of cross-modal alignment. We further introduce a ScMix augmentation strategy that diversifies visual contexts and strengthens both global and local utility of UAPs, thereby reducing reliance on spurious features. In addition, we refine the optimization path by leveraging a temporal hierarchy of historical and estimated future gradients to avoid local minima and stabilize universal perturbation learning. For the text modality, HRA identifies globally influential words by combining intra-sentence and inter-sentence importance measures, and subsequently utilizes these words as universal text perturbations. Extensive experiments across various downstream tasks, VLP models, and datasets demonstrate the superiority of the proposed universal multimodal attacks.

摘要: 现有针对视觉语言预训练（VLP）模型的对抗攻击大多为样本特异性方法，在大规模数据集或新场景中扩展时会产生巨大的计算开销。为克服这一局限，我们提出分层优化攻击（HRA），一种面向VLP模型的通用多模态攻击框架。HRA在样本层面和优化层面同时优化通用对抗扰动（UAPs）。对于图像模态，我们将对抗样本解耦为干净图像和扰动分量，使各分量可独立处理以更有效地破坏跨模态对齐。进一步提出ScMix增强策略，通过多样化视觉上下文来强化UAPs的全局与局部效用，从而降低对伪特征的依赖。此外，我们利用历史梯度与预估未来梯度构建时序层级结构来优化搜索路径，避免陷入局部最优并稳定通用扰动学习过程。对于文本模态，HRA通过融合句内与句间重要性度量来识别全局关键词语，并将其作为通用文本扰动源。在多种下游任务、VLP模型和数据集上的大量实验证明了所提通用多模态攻击方法的优越性。



## **17. Paraphrasing Adversarial Attack on LLM-as-a-Reviewer**

针对LLM作为审稿人的释义对抗攻击 cs.CL

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.06884v1) [paper-pdf](https://arxiv.org/pdf/2601.06884v1)

**Confidence**: 0.95

**Authors**: Masahiro Kaneko

**Abstract**: The use of large language models (LLMs) in peer review systems has attracted growing attention, making it essential to examine their potential vulnerabilities. Prior attacks rely on prompt injection, which alters manuscript content and conflates injection susceptibility with evaluation robustness. We propose the Paraphrasing Adversarial Attack (PAA), a black-box optimization method that searches for paraphrased sequences yielding higher review scores while preserving semantic equivalence and linguistic naturalness. PAA leverages in-context learning, using previous paraphrases and their scores to guide candidate generation. Experiments across five ML and NLP conferences with three LLM reviewers and five attacking models show that PAA consistently increases review scores without changing the paper's claims. Human evaluation confirms that generated paraphrases maintain meaning and naturalness. We also find that attacked papers exhibit increased perplexity in reviews, offering a potential detection signal, and that paraphrasing submissions can partially mitigate attacks.

摘要: 大型语言模型（LLMs）在同行评审系统中的使用日益受到关注，因此有必要审视其潜在脆弱性。现有攻击方法依赖提示注入，这会改变稿件内容，并将注入易感性与评估鲁棒性混为一谈。我们提出释义对抗攻击（PAA），这是一种黑盒优化方法，旨在搜索能产生更高评审分数、同时保持语义等价性和语言自然性的释义序列。PAA利用上下文学习，通过先前的释义及其分数来指导候选生成。在五个ML和NLP会议中，使用三个LLM审稿人和五个攻击模型进行的实验表明，PAA能持续提高评审分数，而不改变论文主张。人工评估证实生成的释义保持了意义和自然性。我们还发现，受攻击论文的评审表现出困惑度增加，这提供了潜在的检测信号，且对提交内容进行释义可部分缓解攻击。



## **18. State Backdoor: Towards Stealthy Real-world Poisoning Attack on Vision-Language-Action Model in State Space**

状态后门：针对状态空间中视觉-语言-动作模型的隐蔽现实世界投毒攻击 cs.CR

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.04266v1) [paper-pdf](https://arxiv.org/pdf/2601.04266v1)

**Confidence**: 0.95

**Authors**: Ji Guo, Wenbo Jiang, Yansong Lin, Yijing Liu, Ruichen Zhang, Guomin Lu, Aiguo Chen, Xinshuo Han, Hongwei Li, Dusit Niyato

**Abstract**: Vision-Language-Action (VLA) models are widely deployed in safety-critical embodied AI applications such as robotics. However, their complex multimodal interactions also expose new security vulnerabilities. In this paper, we investigate a backdoor threat in VLA models, where malicious inputs cause targeted misbehavior while preserving performance on clean data. Existing backdoor methods predominantly rely on inserting visible triggers into visual modality, which suffer from poor robustness and low insusceptibility in real-world settings due to environmental variability. To overcome these limitations, we introduce the State Backdoor, a novel and practical backdoor attack that leverages the robot arm's initial state as the trigger. To optimize trigger for insusceptibility and effectiveness, we design a Preference-guided Genetic Algorithm (PGA) that efficiently searches the state space for minimal yet potent triggers. Extensive experiments on five representative VLA models and five real-world tasks show that our method achieves over 90% attack success rate without affecting benign task performance, revealing an underexplored vulnerability in embodied AI systems.

摘要: 视觉-语言-动作（VLA）模型已广泛应用于机器人等安全关键型具身人工智能应用。然而，其复杂的多模态交互也暴露出新的安全漏洞。本文研究了VLA模型中的后门威胁，即恶意输入会导致目标性错误行为，同时在干净数据上保持正常性能。现有后门方法主要依赖在视觉模态中插入可见触发器，由于环境变化，在现实场景中鲁棒性差且隐蔽性低。为克服这些局限，我们提出了状态后门——一种新颖实用的后门攻击方法，利用机械臂初始状态作为触发器。为优化触发器的隐蔽性和有效性，我们设计了偏好引导遗传算法（PGA），在状态空间中高效搜索最小化但高效的触发器。在五个代表性VLA模型和五个现实任务上的大量实验表明，我们的方法在保持良性任务性能的同时实现了超过90%的攻击成功率，揭示了具身AI系统中一个尚未被充分探索的脆弱性。



## **19. Hidden State Poisoning Attacks against Mamba-based Language Models**

针对基于Mamba的语言模型的隐藏状态投毒攻击 cs.CL

17 pages, 4 figures

**SubmitDate**: 2026-01-06    [abs](http://arxiv.org/abs/2601.01972v2) [paper-pdf](https://arxiv.org/pdf/2601.01972v2)

**Confidence**: 0.95

**Authors**: Alexandre Le Mercier, Chris Develder, Thomas Demeester

**Abstract**: State space models (SSMs) like Mamba offer efficient alternatives to Transformer-based language models, with linear time complexity. Yet, their adversarial robustness remains critically unexplored. This paper studies the phenomenon whereby specific short input phrases induce a partial amnesia effect in such models, by irreversibly overwriting information in their hidden states, referred to as a Hidden State Poisoning Attack (HiSPA). Our benchmark RoBench25 allows evaluating a model's information retrieval capabilities when subject to HiSPAs, and confirms the vulnerability of SSMs against such attacks. Even a recent 52B hybrid SSM-Transformer model from the Jamba family collapses on RoBench25 under optimized HiSPA triggers, whereas pure Transformers do not. We also observe that HiSPA triggers significantly weaken the Jamba model on the popular Open-Prompt-Injections benchmark, unlike pure Transformers. Finally, our interpretability study reveals patterns in Mamba's hidden layers during HiSPAs that could be used to build a HiSPA mitigation system. The full code and data to reproduce the experiments can be found at https://anonymous.4open.science/r/hispa_anonymous-5DB0.

摘要: Mamba等状态空间模型（SSMs）为基于Transformer的语言模型提供了高效的替代方案，具有线性时间复杂度。然而，其对抗鲁棒性仍亟待探索。本文研究了一种现象：特定短输入短语通过不可逆地覆盖模型隐藏状态中的信息，引发部分遗忘效应，称为隐藏状态投毒攻击（HiSPA）。我们提出的基准测试RoBench25可用于评估模型在遭受HiSPA时的信息检索能力，并证实了SSMs对此类攻击的脆弱性。即使是Jamba家族最新的520亿参数混合SSM-Transformer模型，在优化的HiSPA触发词下也会在RoBench25上崩溃，而纯Transformer模型则不会。我们还观察到，与纯Transformer不同，HiSPA触发词会显著削弱Jamba模型在流行的Open-Prompt-Injections基准测试上的表现。最后，我们的可解释性研究揭示了Mamba在HiSPA期间隐藏层的模式，这些模式可用于构建HiSPA缓解系统。完整代码和数据可在https://anonymous.4open.science/r/hispa_anonymous-5DB0获取。



## **20. Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing**

基于LLM的学术评审中的多语言隐藏提示注入攻击 cs.CL

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.23684v1) [paper-pdf](https://arxiv.org/pdf/2512.23684v1)

**Confidence**: 0.95

**Authors**: Panagiotis Theocharopoulos, Ajinkya Kulkarni, Mathew Magimai. -Doss

**Abstract**: Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages.

摘要: 大型语言模型（LLMs）正越来越多地被考虑用于高影响力工作流程，包括学术同行评审。然而，LLMs容易受到文档级隐藏提示注入攻击。在本研究中，我们构建了一个包含约500篇被ICML接受的真实学术论文的数据集，并评估了在这些文档中嵌入隐藏对抗性提示的效果。每篇论文均被注入四种不同语言的语义等效指令，并使用LLM进行评审。我们发现，英语、日语和中文的提示注入会导致评审分数和接受/拒绝决定发生显著变化，而阿拉伯语注入几乎不产生任何影响。这些结果突显了基于LLM的评审系统对文档级提示注入的易感性，并揭示了不同语言间脆弱性的显著差异。



## **21. NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge**

NeuroGenPoisoning：基于遗传优化的外部知识神经元引导攻击，针对大语言模型的检索增强生成 cs.AI

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2510.21144v2) [paper-pdf](https://arxiv.org/pdf/2510.21144v2)

**Confidence**: 0.95

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.

摘要: 检索增强生成（RAG）使大语言模型（LLM）能够在推理过程中动态整合外部知识，从而提高其事实准确性和适应性。然而，攻击者可能注入被污染的外部知识以覆盖模型的内部记忆。现有攻击方法主要通过迭代操纵RAG的检索内容或提示结构，但大多忽略了模型的内部表示动态和神经元级敏感性。RAG污染的根本机制尚未得到充分研究，且未考虑RAG中与强参数化知识冲突的影响。本文提出NeuroGenPoisoning，一种新颖的攻击框架，通过LLM内部神经元归因和遗传优化生成RAG中的对抗性外部知识。该方法首先识别一组“污染响应神经元”，其激活与上下文污染知识高度相关；随后采用遗传算法进化对抗性段落，以最大化激活这些神经元。关键的是，本框架通过观察到的归因信号识别并重用有潜力但初始未成功的外部知识变体，实现了大规模生成有效的污染RAG知识。同时，污染响应神经元引导的污染能有效解决知识冲突。跨模型和数据集的实验结果表明，该方法在保持流畅性的同时，持续实现超过90%的高群体覆盖成功率（POSR）。实证证据表明，本方法能有效解决知识冲突。



## **22. Illusions of Relevance: Arbitrary Content Injection Attacks Deceive Retrievers, Rerankers, and LLM Judges**

相关性幻觉：任意内容注入攻击欺骗检索器、重排序器和LLM评判器 cs.IR

AACL Findings 2025

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2501.18536v2) [paper-pdf](https://arxiv.org/pdf/2501.18536v2)

**Confidence**: 0.95

**Authors**: Manveer Singh Tamber, Jimmy Lin

**Abstract**: This work considers a black-box threat model in which adversaries attempt to propagate arbitrary non-relevant content in search. We show that retrievers, rerankers, and LLM relevance judges are all highly vulnerable to attacks that enable arbitrary content to be promoted to the top of search results and to be assigned perfect relevance scores. We investigate how attackers may achieve this via content injection, injecting arbitrary sentences into relevant passages or query terms into arbitrary passages. Our study analyzes how factors such as model class and size, the balance between relevant and non-relevant content, injection location, toxicity and severity of injected content, and the role of LLM-generated content influence attack success, yielding novel, concerning, and often counterintuitive results. Our results reveal a weakness in embedding models, LLM-based scoring models, and generative LLMs, raising concerns about the general robustness, safety, and trustworthiness of language models regardless of the type of model or the role in which they are employed. We also emphasize the challenges of robust defenses against these attacks. Classifiers and more carefully prompted LLM judges often fail to recognize passages with content injection, especially when considering diverse text topics and styles. Our findings highlight the need for further research into arbitrary content injection attacks. We release our code for further study.

摘要: 本研究探讨了一种黑盒威胁模型，其中攻击者试图在搜索中传播任意不相关内容。我们证明检索器、重排序器和LLM相关性评判器都极易受到攻击，这些攻击能使任意内容被提升至搜索结果顶部并获得完美相关性评分。我们研究了攻击者如何通过内容注入实现这一目标——将任意句子注入相关段落或将查询词注入任意段落。我们的研究分析了模型类别与规模、相关与非相关内容之间的平衡、注入位置、注入内容的毒性和严重程度，以及LLM生成内容的作用等因素如何影响攻击成功率，得出了新颖、令人担忧且往往反直觉的结果。这些结果揭示了嵌入模型、基于LLM的评分模型和生成式LLM的弱点，引发了人们对语言模型整体鲁棒性、安全性和可信度的担忧，无论模型类型或其应用场景如何。我们还强调了针对这些攻击构建稳健防御的挑战。分类器和经过更精心设计的LLM评判器往往无法识别包含内容注入的段落，特别是在考虑多样化文本主题和风格时。我们的发现凸显了对任意内容注入攻击进行进一步研究的必要性。我们已发布代码以供后续研究。



## **23. Transferability of Adversarial Attacks in Video-based MLLMs: A Cross-modal Image-to-Video Approach**

视频多模态大语言模型中对抗攻击的可迁移性：一种跨模态图像到视频方法 cs.CV

**SubmitDate**: 2026-01-09    [abs](http://arxiv.org/abs/2501.01042v4) [paper-pdf](https://arxiv.org/pdf/2501.01042v4)

**Confidence**: 0.95

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Yong-Jie Yin, Bo Han, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models - a common and practical real-world scenario - remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal large language model (I-MLLM) as a surrogate model to craft adversarial video samples. Multimodal interactions and spatiotemporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. Additionally, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as a surrogate model) achieve competitive performance, with average attack success rate (AASR) of 57.98% on MSVD-QA and 58.26% on MSRVTT-QA for Zero-Shot VideoQA tasks, respectively.

摘要: 视频多模态大语言模型（V-MLLMs）在视频-文本多模态任务中已显示出对对抗样本的脆弱性。然而，对抗视频对未见模型的迁移性——这一常见且实际的现实场景——仍未得到探索。本文率先研究了对抗视频样本在V-MLLMs间的迁移性。我们发现现有对抗攻击方法在V-MLLMs的黑盒设置中存在显著局限性，这归因于以下不足：（1）扰动视频特征时缺乏泛化性，（2）仅关注稀疏关键帧，（3）未能整合多模态信息。为克服这些局限并深化对黑盒场景下V-MLLM脆弱性的理解，我们提出了图像到视频多模态大语言模型（I2V-MLLM）攻击。在I2V-MLLM中，我们利用基于图像的多模态大语言模型（I-MLLM）作为代理模型来生成对抗视频样本。通过整合多模态交互和时空信息来破坏潜在空间中的视频表征，从而提升对抗迁移性。此外，引入扰动传播技术以处理不同的未知帧采样策略。实验结果表明，我们的方法能在多个视频-文本多模态任务上生成具有强迁移性的对抗样本，适用于不同V-MLLMs。与这些模型的白盒攻击相比，我们的黑盒攻击（使用BLIP-2作为代理模型）取得了具有竞争力的性能，在Zero-Shot VideoQA任务中，MSVD-QA和MSRVTT-QA的平均攻击成功率（AASR）分别达到57.98%和58.26%。



## **24. On the Adversarial Robustness of 3D Large Vision-Language Models**

关于3D大型视觉语言模型的对抗鲁棒性研究 cs.CV

Under Review

**SubmitDate**: 2026-01-10    [abs](http://arxiv.org/abs/2601.06464v1) [paper-pdf](https://arxiv.org/pdf/2601.06464v1)

**Confidence**: 0.95

**Authors**: Chao Liu, Ngai-Man Cheung

**Abstract**: 3D Vision-Language Models (VLMs), such as PointLLM and GPT4Point, have shown strong reasoning and generalization abilities in 3D understanding tasks. However, their adversarial robustness remains largely unexplored. Prior work in 2D VLMs has shown that the integration of visual inputs significantly increases vulnerability to adversarial attacks, making these models easier to manipulate into generating toxic or misleading outputs. In this paper, we investigate whether incorporating 3D vision similarly compromises the robustness of 3D VLMs. To this end, we present the first systematic study of adversarial robustness in point-based 3D VLMs. We propose two complementary attack strategies: \textit{Vision Attack}, which perturbs the visual token features produced by the 3D encoder and projector to assess the robustness of vision-language alignment; and \textit{Caption Attack}, which directly manipulates output token sequences to evaluate end-to-end system robustness. Each attack includes both untargeted and targeted variants to measure general vulnerability and susceptibility to controlled manipulation. Our experiments reveal that 3D VLMs exhibit significant adversarial vulnerabilities under untargeted attacks, while demonstrating greater resilience against targeted attacks aimed at forcing specific harmful outputs, compared to their 2D counterparts. These findings highlight the importance of improving the adversarial robustness of 3D VLMs, especially as they are deployed in safety-critical applications.

摘要: 3D视觉语言模型（如PointLLM和GPT4Point）在3D理解任务中展现出强大的推理和泛化能力。然而，其对抗鲁棒性在很大程度上尚未得到探索。先前在2D VLM中的研究表明，视觉输入的整合显著增加了对抗攻击的脆弱性，使得这些模型更容易被操纵生成有害或误导性输出。本文研究了3D视觉的引入是否同样会损害3D VLM的鲁棒性。为此，我们首次对基于点云的3D VLM进行了对抗鲁棒性的系统性研究。我们提出了两种互补的攻击策略：\textit{视觉攻击}，通过扰动3D编码器和投影器产生的视觉token特征来评估视觉-语言对齐的鲁棒性；以及\textit{描述攻击}，直接操纵输出token序列以评估端到端系统的鲁棒性。每种攻击都包含无目标和有目标两种变体，分别用于衡量一般脆弱性和受控操纵的易感性。实验表明，3D VLM在无目标攻击下表现出显著的对抗脆弱性，但与2D模型相比，在旨在强制生成特定有害输出的有目标攻击中展现出更强的抵抗力。这些发现凸显了提升3D VLM对抗鲁棒性的重要性，尤其是在部署于安全关键应用场景时。



## **25. FlipLLM: Efficient Bit-Flip Attacks on Multimodal LLMs using Reinforcement Learning**

FlipLLM：基于强化学习的多模态大语言模型高效位翻转攻击 cs.CR

Accepted in IEEE HOST 2026

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09872v1) [paper-pdf](https://arxiv.org/pdf/2512.09872v1)

**Confidence**: 0.95

**Authors**: Khurram Khalil, Khaza Anuarul Hoque

**Abstract**: Generative Artificial Intelligence models, such as Large Language Models (LLMs) and Large Vision Models (VLMs), exhibit state-of-the-art performance but remain vulnerable to hardware-based threats, specifically bit-flip attacks (BFAs). Existing BFA discovery methods lack generalizability and struggle to scale, often failing to analyze the vast parameter space and complex interdependencies of modern foundation models in a reasonable time. This paper proposes FlipLLM, a reinforcement learning (RL) architecture-agnostic framework that formulates BFA discovery as a sequential decision-making problem. FlipLLM combines sensitivity-guided layer pruning with Q-learning to efficiently identify minimal, high-impact bit sets that can induce catastrophic failure. We demonstrate the effectiveness and generalizability of FlipLLM by applying it to a diverse set of models, including prominent text-only LLMs (GPT-2 Large, LLaMA 3.1 8B, and DeepSeek-V2 7B), VLMs such as LLaVA 1.6, and datasets, such as MMLU, MMLU-Pro, VQAv2, and TextVQA. Our results show that FlipLLM can identify critical bits that are vulnerable to BFAs up to 2.5x faster than SOTA methods. We demonstrate that flipping the FlipLLM-identified bits plummets the accuracy of LLaMA 3.1 8B from 69.9% to ~0.2%, and for LLaVA's VQA score from 78% to almost 0%, by flipping as few as 5 and 7 bits, respectively. Further analysis reveals that applying standard hardware protection mechanisms, such as ECC SECDED, to the FlipLLM-identified bit locations completely mitigates the BFA impact, demonstrating the practical value of our framework in guiding hardware-level defenses. FlipLLM offers the first scalable and adaptive methodology for exploring the BFA vulnerability of both language and multimodal foundation models, paving the way for comprehensive hardware-security evaluation.

摘要: 生成式人工智能模型（如大语言模型和大型视觉模型）虽展现出最先进的性能，但仍易受基于硬件的威胁，特别是位翻转攻击。现有BFA发现方法缺乏泛化能力且难以扩展，通常无法在合理时间内分析现代基础模型的庞大参数空间和复杂相互依赖关系。本文提出FlipLLM，一种与架构无关的强化学习框架，将BFA发现建模为序列决策问题。FlipLLM结合敏感度引导的层剪枝与Q学习，高效识别能引发灾难性故障的最小高影响位集合。我们通过将FlipLLM应用于多样化模型（包括主流纯文本LLM、视觉语言模型）和数据集，证明了其有效性和泛化能力。实验表明，FlipLLM识别易受BFA攻击关键位的速度比现有最优方法快达2.5倍。对FlipLLM识别位进行翻转后，LLaMA 3.1 8B的准确率从69.9%骤降至约0.2%，LLaVA的VQA得分从78%降至接近0%，分别仅需翻转5位和7位。进一步分析表明，对FlipLLM识别位应用标准硬件保护机制可完全缓解BFA影响，这证明了我们框架在指导硬件级防御方面的实用价值。FlipLLM为探索语言和多模态基础模型的BFA漏洞提供了首个可扩展的自适应方法，为全面硬件安全评估开辟了新途径。



## **26. Read or Ignore? A Unified Benchmark for Typographic-Attack Robustness and Text Recognition in Vision-Language Models**

读取还是忽略？视觉语言模型中印刷体攻击鲁棒性与文本识别的统一基准 cs.CV

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.11899v1) [paper-pdf](https://arxiv.org/pdf/2512.11899v1)

**Confidence**: 0.95

**Authors**: Futa Waseda, Shojiro Yamabe, Daiki Shiono, Kento Sasaki, Tsubasa Takahashi

**Abstract**: Large vision-language models (LVLMs) are vulnerable to typographic attacks, where misleading text within an image overrides visual understanding. Existing evaluation protocols and defenses, largely focused on object recognition, implicitly encourage ignoring text to achieve robustness; however, real-world scenarios often require joint reasoning over both objects and text (e.g., recognizing pedestrians while reading traffic signs). To address this, we introduce a novel task, Read-or-Ignore VQA (RIO-VQA), which formalizes selective text use in visual question answering (VQA): models must decide, from context, when to read text and when to ignore it. For evaluation, we present the Read-or-Ignore Benchmark (RIO-Bench), a standardized dataset and protocol that, for each real image, provides same-scene counterfactuals (read / ignore) by varying only the textual content and question type. Using RIO-Bench, we show that strong LVLMs and existing defenses fail to balance typographic robustness and text-reading capability, highlighting the need for improved approaches. Finally, RIO-Bench enables a novel data-driven defense that learns adaptive selective text use, moving beyond prior non-adaptive, text-ignoring defenses. Overall, this work reveals a fundamental misalignment between the existing evaluation scope and real-world requirements, providing a principled path toward reliable LVLMs. Our Project Page is at https://turingmotors.github.io/rio-vqa/.

摘要: 大型视觉语言模型（LVLMs）容易受到印刷体攻击的影响，即图像中的误导性文本会覆盖视觉理解。现有的评估协议和防御方法主要关注物体识别，隐含地鼓励忽略文本以实现鲁棒性；然而，现实场景通常需要对物体和文本进行联合推理（例如，识别行人的同时读取交通标志）。为解决这一问题，我们引入了一项新任务——读取或忽略视觉问答（RIO-VQA），该任务形式化了视觉问答（VQA）中文本的选择性使用：模型必须根据上下文决定何时读取文本、何时忽略文本。为进行评估，我们提出了读取或忽略基准（RIO-Bench），这是一个标准化的数据集和协议，为每张真实图像提供仅通过改变文本内容和问题类型生成的同场景反事实（读取/忽略）样本。使用RIO-Bench，我们发现强大的LVLMs和现有防御方法均无法平衡印刷体攻击鲁棒性与文本读取能力，这凸显了改进方法的必要性。最后，RIO-Bench支持一种新颖的数据驱动防御方法，通过学习自适应选择性文本使用，超越了先前非自适应的文本忽略防御策略。总体而言，这项工作揭示了现有评估范围与现实需求之间的根本性错位，为构建可靠的LVLMs提供了原则性路径。项目页面位于：https://turingmotors.github.io/rio-vqa/。



## **27. When Ads Become Profiles: Uncovering the Invisible Risk of Web Advertising at Scale with LLMs**

当广告成为用户画像：利用LLM大规模揭示网络广告的隐形风险 cs.HC

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2509.18874v2) [paper-pdf](https://arxiv.org/pdf/2509.18874v2)

**Confidence**: 0.95

**Authors**: Baiyu Chen, Benjamin Tag, Hao Xue, Daniel Angus, Flora Salim

**Abstract**: Regulatory limits on explicit targeting have not eliminated algorithmic profiling on the Web, as optimisation systems still adapt ad delivery to users' private attributes. The widespread availability of powerful zero-shot multimodal Large Language Models (LLMs) has dramatically lowered the barrier for exploiting these latent signals for adversarial inference. We investigate this emerging societal risk, specifically how adversaries can now exploit these signals to reverse-engineer private attributes from ad exposure alone. We introduce a novel pipeline that leverages LLMs as adversarial inference engines to perform natural language profiling. Applying this method to a longitudinal dataset comprising over 435,000 ad impressions collected from 891 users, we conducted a large-scale study to assess the feasibility and precision of inferring private attributes from passive online ad observations. Our results demonstrate that off-the-shelf LLMs can accurately reconstruct complex user private attributes, including party preference, employment status, and education level, consistently outperforming strong census-based priors and matching or exceeding human social perception, while operating at only a fraction of the cost (223$\times$ lower) and time (52$\times$ faster) required by humans. Critically, actionable profiling is feasible even within short observation windows, indicating that prolonged tracking is not a prerequisite for a successful attack. These findings provide the first empirical evidence that ad streams serve as a high-fidelity digital footprint, enabling off-platform profiling that inherently bypasses current platform safeguards, highlighting a systemic vulnerability in the ad ecosystem and the urgent need for responsible web AI governance in the generative AI era. The code is available at https://github.com/Breezelled/when-ads-become-profiles.

摘要: 尽管监管对显式定向广告设限，但网络上的算法画像并未消除——优化系统仍会根据用户的私有属性调整广告投放。强大零样本多模态大语言模型（LLMs）的广泛普及，极大降低了利用这些潜在信号进行对抗性推断的门槛。我们研究了这一新兴社会风险，特别是攻击者如何仅通过广告曝光逆向推断私有属性。我们提出一种新颖的流程，利用LLMs作为对抗性推断引擎进行自然语言画像分析。将该方法应用于包含891名用户超过435,000条广告展示的纵向数据集，我们开展了大规模研究，评估从被动在线广告观察中推断私有属性的可行性与精确度。结果表明，现成的LLMs能准确重构复杂的用户私有属性（包括政党偏好、就业状况和教育水平），持续优于基于人口普查的强先验基准，达到或超越人类社交感知能力，而成本仅为人类的1/223，速度提升52倍。关键的是，即使在短期观察窗口内也能实现有效的画像分析，表明长期追踪并非攻击成功的必要条件。这些发现首次提供实证证据：广告流可作为高保真数字足迹，实现绕开现有平台防护措施的跨平台画像，揭示了广告生态的系统性漏洞，以及生成式AI时代负责任网络AI治理的迫切需求。代码已开源：https://github.com/Breezelled/when-ads-become-profiles。



## **28. When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models**

当机器人服从补丁：针对视觉-语言-动作模型的通用可迁移补丁攻击 cs.CV

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2511.21192v2) [paper-pdf](https://arxiv.org/pdf/2511.21192v2)

**Confidence**: 0.95

**Authors**: Hui Lu, Yi Yu, Yiming Yang, Chenyu Yi, Qixin Zhang, Bingquan Shen, Alex C. Kot, Xudong Jiang

**Abstract**: Vision-Language-Action (VLA) models are vulnerable to adversarial attacks, yet universal and transferable attacks remain underexplored, as most existing patches overfit to a single model and fail in black-box settings. To address this gap, we present a systematic study of universal, transferable adversarial patches against VLA-driven robots under unknown architectures, finetuned variants, and sim-to-real shifts. We introduce UPA-RFAS (Universal Patch Attack via Robust Feature, Attention, and Semantics), a unified framework that learns a single physical patch in a shared feature space while promoting cross-model transfer. UPA-RFAS combines (i) a feature-space objective with an $\ell_1$ deviation prior and repulsive InfoNCE loss to induce transferable representation shifts, (ii) a robustness-augmented two-phase min-max procedure where an inner loop learns invisible sample-wise perturbations and an outer loop optimizes the universal patch against this hardened neighborhood, and (iii) two VLA-specific losses: Patch Attention Dominance to hijack text$\to$vision attention and Patch Semantic Misalignment to induce image-text mismatch without labels. Experiments across diverse VLA models, manipulation suites, and physical executions show that UPA-RFAS consistently transfers across models, tasks, and viewpoints, exposing a practical patch-based attack surface and establishing a strong baseline for future defenses.

摘要: 视觉-语言-动作（VLA）模型易受对抗性攻击，但通用且可迁移的攻击仍研究不足，现有补丁大多过拟合单一模型且在黑盒设置中失效。为填补这一空白，我们系统研究了针对未知架构、微调变体及仿真到现实迁移的VLA驱动机器人的通用可迁移对抗补丁。我们提出UPA-RFAS（基于鲁棒特征、注意力与语义的通用补丁攻击），这是一个在共享特征空间中学习单一物理补丁并促进跨模型迁移的统一框架。UPA-RFAS结合：（i）采用ℓ1偏差先验与排斥性InfoNCE损失的特征空间目标，以诱导可迁移的表征偏移；（ii）鲁棒性增强的两阶段极小-极大过程：内循环学习不可见的样本级扰动，外循环针对该强化邻域优化通用补丁；（iii）两种VLA专用损失：补丁注意力主导（用于劫持文本→视觉注意力）与补丁语义错配（无需标签即可诱导图文失配）。跨多种VLA模型、操作套件及物理执行的实验表明，UPA-RFAS能持续跨模型、任务和视角迁移，揭示了基于补丁的实用攻击面，并为未来防御建立了强基准。



## **29. Universal Adversarial Suffixes Using Calibrated Gumbel-Softmax Relaxation**

基于校准Gumbel-Softmax松弛的通用对抗后缀 cs.CL

10 pages

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08123v1) [paper-pdf](https://arxiv.org/pdf/2512.08123v1)

**Confidence**: 0.95

**Authors**: Sampriti Soor, Suklav Ghosh, Arijit Sur

**Abstract**: Language models (LMs) are often used as zero-shot or few-shot classifiers by scoring label words, but they remain fragile to adversarial prompts. Prior work typically optimizes task- or model-specific triggers, making results difficult to compare and limiting transferability. We study universal adversarial suffixes: short token sequences (4-10 tokens) that, when appended to any input, broadly reduce accuracy across tasks and models. Our approach learns the suffix in a differentiable "soft" form using Gumbel-Softmax relaxation and then discretizes it for inference. Training maximizes calibrated cross-entropy on the label region while masking gold tokens to prevent trivial leakage, with entropy regularization to avoid collapse. A single suffix trained on one model transfers effectively to others, consistently lowering both accuracy and calibrated confidence. Experiments on sentiment analysis, natural language inference, paraphrase detection, commonsense QA, and physical reasoning with Qwen2-1.5B, Phi-1.5, and TinyLlama-1.1B demonstrate consistent attack effectiveness and transfer across tasks and model families.

摘要: 语言模型常被用作零样本或少样本分类器，通过为标签词打分实现分类，但其对对抗性提示仍显脆弱。先前研究通常针对特定任务或模型优化触发词，导致结果难以比较且可迁移性有限。本文研究通用对抗后缀：即短令牌序列（4-10个令牌），当附加到任意输入时，能广泛降低跨任务和跨模型的准确率。我们的方法采用Gumbel-Softmax松弛技术学习可微分的“软”形式后缀，随后在推理时将其离散化。训练过程通过掩码黄金令牌防止信息泄露，最大化标签区域的校准交叉熵，并引入熵正则化避免坍缩。在单一模型上训练的后缀能有效迁移至其他模型，持续降低准确率和校准置信度。在情感分析、自然语言推理、复述检测、常识问答和物理推理任务上，使用Qwen2-1.5B、Phi-1.5和TinyLlama-1.1B模型的实验表明，该方法具有稳定的攻击效果和跨任务、跨模型家族的迁移能力。



## **30. Survey of Adversarial Robustness in Multimodal Large Language Models**

多模态大语言模型对抗鲁棒性研究综述 cs.CV

9 pages

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13962v1) [paper-pdf](https://arxiv.org/pdf/2503.13962v1)

**Confidence**: 0.95

**Authors**: Chengze Jiang, Zhuangzhuang Wang, Minjing Dong, Jie Gui

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance in artificial intelligence by facilitating integrated understanding across diverse modalities, including text, images, video, audio, and speech. However, their deployment in real-world applications raises significant concerns about adversarial vulnerabilities that could compromise their safety and reliability. Unlike unimodal models, MLLMs face unique challenges due to the interdependencies among modalities, making them susceptible to modality-specific threats and cross-modal adversarial manipulations. This paper reviews the adversarial robustness of MLLMs, covering different modalities. We begin with an overview of MLLMs and a taxonomy of adversarial attacks tailored to each modality. Next, we review key datasets and evaluation metrics used to assess the robustness of MLLMs. After that, we provide an in-depth review of attacks targeting MLLMs across different modalities. Our survey also identifies critical challenges and suggests promising future research directions.

摘要: 多模态大语言模型（MLLMs）通过促进跨文本、图像、视频、音频和语音等多种模态的集成理解，在人工智能领域展现出卓越性能。然而，其在现实应用中的部署引发了对其对抗脆弱性的重大关切，这些脆弱性可能损害其安全性和可靠性。与单模态模型不同，MLLMs因模态间相互依赖性面临独特挑战，使其易受模态特定威胁和跨模态对抗操纵的影响。本文综述了MLLMs的对抗鲁棒性研究，涵盖不同模态。首先概述MLLMs及针对各模态的对抗攻击分类体系，随后回顾用于评估MLLMs鲁棒性的关键数据集和评价指标，继而深入评述针对不同模态MLLMs的攻击方法。本综述还识别了关键挑战，并提出了未来有前景的研究方向。



## **31. Securing LLM-as-a-Service for Small Businesses: An Industry Case Study of a Distributed Chatbot Deployment Platform**

为中小企业保障LLM即服务安全：分布式聊天机器人部署平台的行业案例研究 cs.DC

Accepted by AISC 2026

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15528v1) [paper-pdf](https://arxiv.org/pdf/2601.15528v1)

**Confidence**: 0.90

**Authors**: Jiazhu Xie, Bowen Li, Heyu Fu, Chong Gao, Ziqi Xu, Fengling Han

**Abstract**: Large Language Model (LLM)-based question-answering systems offer significant potential for automating customer support and internal knowledge access in small businesses, yet their practical deployment remains challenging due to infrastructure costs, engineering complexity, and security risks, particularly in retrieval-augmented generation (RAG)-based settings. This paper presents an industry case study of an open-source, multi-tenant platform that enables small businesses to deploy customised LLM-based support chatbots via a no-code workflow. The platform is built on distributed, lightweight k3s clusters spanning heterogeneous, low-cost machines and interconnected through an encrypted overlay network, enabling cost-efficient resource pooling while enforcing container-based isolation and per-tenant data access controls. In addition, the platform integrates practical, platform-level defences against prompt injection attacks in RAG-based chatbots, translating insights from recent prompt injection research into deployable security mechanisms without requiring model retraining or enterprise-scale infrastructure. We evaluate the proposed platform through a real-world e-commerce deployment, demonstrating that secure and efficient LLM-based chatbot services can be achieved under realistic cost, operational, and security constraints faced by small businesses.

摘要: 基于大语言模型（LLM）的问答系统为中小企业自动化客户支持和内部知识访问提供了巨大潜力，但由于基础设施成本、工程复杂性和安全风险（尤其在基于检索增强生成（RAG）的场景中），其实际部署仍面临挑战。本文介绍了一个开源多租户平台的行业案例研究，该平台使中小企业能够通过无代码工作流部署定制的基于LLM的支持聊天机器人。该平台构建在分布式轻量级k3s集群之上，跨越异构低成本机器，并通过加密覆盖网络互连，实现了成本效益的资源池化，同时强制执行基于容器的隔离和按租户数据访问控制。此外，该平台集成了针对RAG聊天机器人中提示注入攻击的实用平台级防御措施，将近期提示注入研究的见解转化为可部署的安全机制，无需模型重新训练或企业级基础设施。我们通过一个真实电子商务部署评估了所提出的平台，证明在中小企业面临的现实成本、运营和安全约束下，可以实现安全高效的基于LLM的聊天机器人服务。



## **32. SilentDrift: Exploiting Action Chunking for Stealthy Backdoor Attacks on Vision-Language-Action Models**

SilentDrift：利用动作分块对视觉-语言-动作模型实施隐蔽后门攻击 cs.CR

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.14323v1) [paper-pdf](https://arxiv.org/pdf/2601.14323v1)

**Confidence**: 0.90

**Authors**: Bingxin Xu, Yuzhang Shang, Binghui Wang, Emilio Ferrara

**Abstract**: Vision-Language-Action (VLA) models are increasingly deployed in safety-critical robotic applications, yet their security vulnerabilities remain underexplored. We identify a fundamental security flaw in modern VLA systems: the combination of action chunking and delta pose representations creates an intra-chunk visual open-loop. This mechanism forces the robot to execute K-step action sequences, allowing per-step perturbations to accumulate through integration. We propose SILENTDRIFT, a stealthy black-box backdoor attack exploiting this vulnerability. Our method employs the Smootherstep function to construct perturbations with guaranteed C2 continuity, ensuring zero velocity and acceleration at trajectory boundaries to satisfy strict kinematic consistency constraints. Furthermore, our keyframe attack strategy selectively poisons only the critical approach phase, maximizing impact while minimizing trigger exposure. The resulting poisoned trajectories are visually indistinguishable from successful demonstrations. Evaluated on the LIBERO, SILENTDRIFT achieves a 93.2% Attack Success Rate with a poisoning rate under 2%, while maintaining a 95.3% Clean Task Success Rate.

摘要: 视觉-语言-动作（VLA）模型正日益部署于安全关键型机器人应用中，但其安全漏洞仍未得到充分探索。我们发现现代VLA系统存在一个根本性安全缺陷：动作分块与增量位姿表示的结合形成了分块内视觉开环机制。该机制强制机器人执行K步动作序列，使得逐步扰动通过积分不断累积。我们提出SILENTDRIFT——一种利用此漏洞的隐蔽黑盒后门攻击方法。该方法采用Smootherstep函数构建具有C2连续性保证的扰动，确保轨迹边界处的速度与加速度为零，以满足严格的运动学一致性约束。此外，我们的关键帧攻击策略选择性地仅毒化关键接近阶段，在最大化攻击影响的同时最小化触发器暴露。生成的毒化轨迹在视觉上与成功演示无法区分。在LIBERO基准上的评估显示，SILENTDRIFT在低于2%的毒化率下实现了93.2%的攻击成功率，同时保持95.3%的清洁任务成功率。



## **33. Evaluation of Hate Speech Detection Using Large Language Models and Geographical Contextualization**

基于大语言模型与地理情境化的仇恨言论检测评估 cs.CL

6 pages, 2 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19612v1) [paper-pdf](https://arxiv.org/pdf/2502.19612v1)

**Confidence**: 0.90

**Authors**: Anwar Hossain Zahid, Monoshi Kumar Roy, Swarna Das

**Abstract**: The proliferation of hate speech on social media is one of the serious issues that is bringing huge impacts to society: an escalation of violence, discrimination, and social fragmentation. The problem of detecting hate speech is intrinsically multifaceted due to cultural, linguistic, and contextual complexities and adversarial manipulations. In this study, we systematically investigate the performance of LLMs on detecting hate speech across multilingual datasets and diverse geographic contexts. Our work presents a new evaluation framework in three dimensions: binary classification of hate speech, geography-aware contextual detection, and robustness to adversarially generated text. Using a dataset of 1,000 comments from five diverse regions, we evaluate three state-of-the-art LLMs: Llama2 (13b), Codellama (7b), and DeepSeekCoder (6.7b). Codellama had the best binary classification recall with 70.6% and an F1-score of 52.18%, whereas DeepSeekCoder had the best performance in geographic sensitivity, correctly detecting 63 out of 265 locations. The tests for adversarial robustness also showed significant weaknesses; Llama2 misclassified 62.5% of manipulated samples. These results bring to light the trade-offs between accuracy, contextual understanding, and robustness in the current versions of LLMs. This work has thus set the stage for developing contextually aware, multilingual hate speech detection systems by underlining key strengths and limitations, therefore offering actionable insights for future research and real-world applications.

摘要: 社交媒体上仇恨言论的泛滥已成为严重的社会问题，导致暴力升级、歧视加剧和社会分裂。由于文化、语言、情境复杂性及对抗性操纵，仇恨言论检测本质上具有多面性。本研究系统评估了大语言模型在多语言数据集及不同地理情境下的仇恨言论检测性能。我们提出了一个三维评估框架：仇恨言论的二元分类、地理感知的情境检测，以及对对抗生成文本的鲁棒性。基于来自五个不同地区的1000条评论数据集，我们评估了三种先进大语言模型：Llama2（13b）、Codellama（7b）和DeepSeekCoder（6.7b）。Codellama在二元分类召回率上表现最佳（70.6%），F1分数为52.18%；而DeepSeekCoder在地理敏感性检测中表现最优，在265个位置中正确识别了63个。对抗鲁棒性测试显示出显著缺陷：Llama2对62.5%的操纵样本产生误判。这些结果揭示了当前大语言模型在准确性、情境理解与鲁棒性之间的权衡。本研究通过明确关键优势与局限，为开发情境感知的多语言仇恨言论检测系统奠定了基础，并为未来研究及实际应用提供了可行见解。



## **34. Robust Fake News Detection using Large Language Models under Adversarial Sentiment Attacks**

基于大语言模型的鲁棒虚假新闻检测：应对对抗性情感攻击 cs.CL

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15277v1) [paper-pdf](https://arxiv.org/pdf/2601.15277v1)

**Confidence**: 0.85

**Authors**: Sahar Tahmasebi, Eric Müller-Budack, Ralph Ewerth

**Abstract**: Misinformation and fake news have become a pressing societal challenge, driving the need for reliable automated detection methods. Prior research has highlighted sentiment as an important signal in fake news detection, either by analyzing which sentiments are associated with fake news or by using sentiment and emotion features for classification. However, this poses a vulnerability since adversaries can manipulate sentiment to evade detectors especially with the advent of large language models (LLMs). A few studies have explored adversarial samples generated by LLMs, but they mainly focus on stylistic features such as writing style of news publishers. Thus, the crucial vulnerability of sentiment manipulation remains largely unexplored. In this paper, we investigate the robustness of state-of-the-art fake news detectors under sentiment manipulation. We introduce AdSent, a sentiment-robust detection framework designed to ensure consistent veracity predictions across both original and sentiment-altered news articles. Specifically, we (1) propose controlled sentiment-based adversarial attacks using LLMs, (2) analyze the impact of sentiment shifts on detection performance. We show that changing the sentiment heavily impacts the performance of fake news detection models, indicating biases towards neutral articles being real, while non-neutral articles are often classified as fake content. (3) We introduce a novel sentiment-agnostic training strategy that enhances robustness against such perturbations. Extensive experiments on three benchmark datasets demonstrate that AdSent significantly outperforms competitive baselines in both accuracy and robustness, while also generalizing effectively to unseen datasets and adversarial scenarios.

摘要: 虚假信息和假新闻已成为紧迫的社会挑战，推动了对可靠自动化检测方法的需求。先前研究强调情感是假新闻检测中的重要信号，包括分析假新闻相关的情感特征或利用情感特征进行分类。然而，这也带来了脆弱性，因为攻击者可能操纵情感以逃避检测器，尤其是在大语言模型（LLMs）兴起后。少数研究探索了LLMs生成的对抗样本，但主要关注新闻发布者的写作风格等文体特征。因此，情感操纵这一关键脆弱性尚未得到充分探索。本文研究了最先进的假新闻检测器在情感操纵下的鲁棒性。我们提出了AdSent——一种情感鲁棒检测框架，旨在确保对原始和情感修改后的新闻文章保持一致的真相预测。具体而言，我们（1）提出使用LLMs进行基于情感的可控对抗攻击，（2）分析情感偏移对检测性能的影响。实验表明，改变情感会严重影响假新闻检测模型的性能，揭示模型存在偏向性：中性文章易被判定为真实，而非中性文章常被归类为虚假内容。（3）我们提出了一种新颖的情感无关训练策略，以增强对此类扰动的鲁棒性。在三个基准数据集上的大量实验表明，AdSent在准确性和鲁棒性上均显著优于竞争基线，同时能有效泛化到未见数据集和对抗场景。



## **35. Multimodal Generative Engine Optimization: Rank Manipulation for Vision-Language Model Rankers**

多模态生成引擎优化：针对视觉语言模型排序器的排名操纵 cs.CL

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2601.12263v1) [paper-pdf](https://arxiv.org/pdf/2601.12263v1)

**Confidence**: 0.85

**Authors**: Yixuan Du, Chenxiao Yu, Haoyan Xu, Ziyi Wang, Yue Zhao, Xiyang Hu

**Abstract**: Vision-Language Models (VLMs) are rapidly replacing unimodal encoders in modern retrieval and recommendation systems. While their capabilities are well-documented, their robustness against adversarial manipulation in competitive ranking scenarios remains largely unexplored. In this paper, we uncover a critical vulnerability in VLM-based product search: multimodal ranking attacks. We present Multimodal Generative Engine Optimization (MGEO), a novel adversarial framework that enables a malicious actor to unfairly promote a target product by jointly optimizing imperceptible image perturbations and fluent textual suffixes. Unlike existing attacks that treat modalities in isolation, MGEO employs an alternating gradient-based optimization strategy to exploit the deep cross-modal coupling within the VLM. Extensive experiments on real-world datasets using state-of-the-art models demonstrate that our coordinated attack significantly outperforms text-only and image-only baselines. These findings reveal that multimodal synergy, typically a strength of VLMs, can be weaponized to compromise the integrity of search rankings without triggering conventional content filters.

摘要: 视觉语言模型（VLMs）正迅速取代现代检索与推荐系统中的单模态编码器。尽管其能力已有充分记录，但在竞争性排序场景下对抗对抗性操纵的鲁棒性仍基本未被探索。本文揭示了基于VLM的产品搜索中存在一个关键漏洞：多模态排序攻击。我们提出了多模态生成引擎优化（MGEO），这是一种新颖的对抗性框架，使恶意行为者能够通过联合优化难以察觉的图像扰动和流畅的文本后缀，不公平地提升目标产品排名。与现有孤立处理各模态的攻击不同，MGEO采用交替梯度优化策略，利用VLM内部深度的跨模态耦合机制。基于真实世界数据集和最先进模型的大量实验表明，我们的协同攻击显著优于纯文本和纯图像基线方法。这些发现揭示，多模态协同效应——通常被视为VLMs的优势——可能被武器化，从而在不触发传统内容过滤器的情况下破坏搜索排名的完整性。



## **36. STEAD: Robust Provably Secure Linguistic Steganography with Diffusion Language Model**

STEAD：基于扩散语言模型的鲁棒可证明安全语言隐写术 cs.CR

NeurIPS 2025 poster

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.14778v1) [paper-pdf](https://arxiv.org/pdf/2601.14778v1)

**Confidence**: 0.85

**Authors**: Yuang Qi, Na Zhao, Qiyi Yao, Benlong Wu, Weiming Zhang, Nenghai Yu, Kejiang Chen

**Abstract**: Recent provably secure linguistic steganography (PSLS) methods rely on mainstream autoregressive language models (ARMs) to address historically challenging tasks, that is, to disguise covert communication as ``innocuous'' natural language communication. However, due to the characteristic of sequential generation of ARMs, the stegotext generated by ARM-based PSLS methods will produce serious error propagation once it changes, making existing methods unavailable under an active tampering attack. To address this, we propose a robust, provably secure linguistic steganography with diffusion language models (DLMs). Unlike ARMs, DLMs can generate text in a partially parallel manner, allowing us to find robust positions for steganographic embedding that can be combined with error-correcting codes. Furthermore, we introduce error correction strategies, including pseudo-random error correction and neighborhood search correction, during steganographic extraction. Theoretical proof and experimental results demonstrate that our method is secure and robust. It can resist token ambiguity in stegotext segmentation and, to some extent, withstand token-level attacks of insertion, deletion, and substitution.

摘要: 近期可证明安全语言隐写术（PSLS）方法依赖主流自回归语言模型（ARM）来解决历史性难题——将隐蔽通信伪装为“无害”的自然语言通信。然而，由于ARM顺序生成的特点，基于ARM的PSLS方法生成的隐写文本一旦被修改会产生严重的错误传播，使得现有方法在主动篡改攻击下失效。为此，我们提出一种基于扩散语言模型（DLM）的鲁棒可证明安全语言隐写术。与ARM不同，DLM能以部分并行方式生成文本，使我们能够找到可与纠错码结合的鲁棒隐写嵌入位置。此外，我们在隐写提取阶段引入了纠错策略，包括伪随机纠错和邻域搜索纠错。理论证明和实验结果表明，我们的方法具有安全性和鲁棒性，能够抵抗隐写文本分词中的标记歧义，并在一定程度上抵御插入、删除和替换等标记级攻击。



## **37. VizDefender: Unmasking Visualization Tampering through Proactive Localization and Intent Inference**

VizDefender：通过主动定位与意图推断揭示可视化篡改 cs.CV

IEEE Transactions on Visualization and Computer Graphics (IEEE PacificVis'26 TVCG Track)

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18853v1) [paper-pdf](https://arxiv.org/pdf/2512.18853v1)

**Confidence**: 0.85

**Authors**: Sicheng Song, Yanjie Zhang, Zixin Chen, Huamin Qu, Changbo Wang, Chenhui Li

**Abstract**: The integrity of data visualizations is increasingly threatened by image editing techniques that enable subtle yet deceptive tampering. Through a formative study, we define this challenge and categorize tampering techniques into two primary types: data manipulation and visual encoding manipulation. To address this, we present VizDefender, a framework for tampering detection and analysis. The framework integrates two core components: 1) a semi-fragile watermark module that protects the visualization by embedding a location map to images, which allows for the precise localization of tampered regions while preserving visual quality, and 2) an intent analysis module that leverages Multimodal Large Language Models (MLLMs) to interpret manipulation, inferring the attacker's intent and misleading effects. Extensive evaluations and user studies demonstrate the effectiveness of our methods.

摘要: 数据可视化的完整性正日益受到图像编辑技术的威胁，这些技术能够实现隐蔽而具有欺骗性的篡改。通过一项形成性研究，我们界定了这一挑战，并将篡改技术主要分为两类：数据操纵和视觉编码操纵。为此，我们提出了VizDefender，一个用于篡改检测和分析的框架。该框架整合了两个核心组件：1）一个半脆弱水印模块，通过向图像嵌入位置图来保护可视化，能够在保持视觉质量的同时精确定位篡改区域；2）一个意图分析模块，利用多模态大语言模型（MLLMs）解读篡改行为，推断攻击者意图及误导效果。广泛的评估和用户研究证明了我们方法的有效性。



## **38. Boosting RL-Based Visual Reasoning with Selective Adversarial Entropy Intervention**

通过选择性对抗熵干预增强基于强化学习的视觉推理能力 cs.AI

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10414v1) [paper-pdf](https://arxiv.org/pdf/2512.10414v1)

**Confidence**: 0.85

**Authors**: Yang Yu, Zhuangzhuang Chen, Siqi Wang, Lanqing Li, Xiaomeng Li

**Abstract**: Recently, reinforcement learning (RL) has become a common choice in enhancing the reasoning capabilities of vision-language models (VLMs). Considering existing RL-based finetuning methods, entropy intervention turns out to be an effective way to benefit exploratory ability, thereby improving policy performance. Notably, most existing studies intervene in entropy by simply controlling the update of specific tokens during policy optimization of RL. They ignore the entropy intervention during the RL sampling that can boost the performance of GRPO by improving the diversity of responses. In this paper, we propose Selective-adversarial Entropy Intervention, namely SaEI, which enhances policy entropy by distorting the visual input with the token-selective adversarial objective coming from the entropy of sampled responses. Specifically, we first propose entropy-guided adversarial sampling (EgAS) that formulates the entropy of sampled responses as an adversarial objective. Then, the corresponding adversarial gradient can be used to attack the visual input for producing adversarial samples, allowing the policy model to explore a larger answer space during RL sampling. Then, we propose token-selective entropy computation (TsEC) to maximize the effectiveness of adversarial attack in EgAS without distorting factual knowledge within VLMs. Extensive experiments on both in-domain and out-of-domain datasets show that our proposed method can greatly improve policy exploration via entropy intervention, to boost reasoning capabilities. Code will be released once the paper is accepted.

摘要: 近年来，强化学习（RL）已成为增强视觉语言模型（VLMs）推理能力的常用方法。在现有的基于RL的微调方法中，熵干预被证明是提升探索能力从而改善策略性能的有效途径。值得注意的是，现有研究大多通过在RL策略优化过程中简单控制特定token的更新来进行熵干预，却忽略了RL采样阶段的熵干预——这能够通过提升响应多样性来增强GRPO的性能。本文提出选择性对抗熵干预方法（SaEI），该方法通过利用采样响应的熵构建token选择性对抗目标来扭曲视觉输入，从而增强策略熵。具体而言，我们首先提出熵引导对抗采样（EgAS），将采样响应的熵构建为对抗目标。随后，相应的对抗梯度可用于攻击视觉输入以生成对抗样本，使策略模型在RL采样过程中能够探索更大的答案空间。接着，我们提出token选择性熵计算（TsEC），在不扭曲VLMs内部事实知识的前提下，最大化EgAS中对抗攻击的有效性。在领域内和领域外数据集上的大量实验表明，我们提出的方法能够通过熵干预显著提升策略探索能力，从而增强推理性能。代码将在论文录用后发布。



## **39. Explainable Adversarial-Robust Vision-Language-Action Model for Robotic Manipulation**

可解释对抗鲁棒的视觉-语言-动作模型在机器人操作中的应用 cs.CV

Accepted to MobieSec 2025 (poster session)

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.11865v1) [paper-pdf](https://arxiv.org/pdf/2512.11865v1)

**Confidence**: 0.85

**Authors**: Ju-Young Kim, Ji-Hong Park, Myeongjun Kim, Gun-Woo Kim

**Abstract**: Smart farming has emerged as a key technology for advancing modern agriculture through automation and intelligent control. However, systems relying on RGB cameras for perception and robotic manipulators for control, common in smart farming, are vulnerable to photometric perturbations such as hue, illumination, and noise changes, which can cause malfunction under adversarial attacks. To address this issue, we propose an explainable adversarial-robust Vision-Language-Action model based on the OpenVLA-OFT framework. The model integrates an Evidence-3 module that detects photometric perturbations and generates natural language explanations of their causes and effects. Experiments show that the proposed model reduces Current Action L1 loss by 21.7% and Next Actions L1 loss by 18.4% compared to the baseline, demonstrating improved action prediction accuracy and explainability under adversarial conditions.

摘要: 智慧农业已成为通过自动化和智能控制推进现代农业的关键技术。然而，依赖RGB摄像头进行感知和机器人操作臂进行控制的系统（在智慧农业中常见）容易受到色调、光照和噪声变化等光度扰动的攻击，这些扰动可能导致系统在对抗攻击下发生故障。为解决这一问题，我们基于OpenVLA-OFT框架提出了一种可解释对抗鲁棒的视觉-语言-动作模型。该模型集成了Evidence-3模块，能够检测光度扰动并生成关于其成因和影响的自然语言解释。实验表明，与基线模型相比，所提模型将当前动作L1损失降低了21.7%，将下一动作L1损失降低了18.4%，证明了在对抗条件下动作预测准确性和可解释性的提升。



## **40. Skeletonization-Based Adversarial Perturbations on Large Vision Language Model's Mathematical Text Recognition**

基于骨架化的对抗性扰动在大型视觉语言模型数学文本识别中的应用 cs.CV

accepted to ITC-CSCC 2025

**SubmitDate**: 2026-01-08    [abs](http://arxiv.org/abs/2601.04752v1) [paper-pdf](https://arxiv.org/pdf/2601.04752v1)

**Confidence**: 0.85

**Authors**: Masatomo Yoshida, Haruto Namura, Nicola Adami, Masahiro Okuda

**Abstract**: This work explores the visual capabilities and limitations of foundation models by introducing a novel adversarial attack method utilizing skeletonization to reduce the search space effectively. Our approach specifically targets images containing text, particularly mathematical formula images, which are more challenging due to their LaTeX conversion and intricate structure. We conduct a detailed evaluation of both character and semantic changes between original and adversarially perturbed outputs to provide insights into the models' visual interpretation and reasoning abilities. The effectiveness of our method is further demonstrated through its application to ChatGPT, which shows its practical implications in real-world scenarios.

摘要: 本研究通过引入一种利用骨架化有效缩减搜索空间的新型对抗攻击方法，探索基础模型的视觉能力与局限性。我们的方法专门针对包含文本的图像，特别是数学公式图像——这类图像因其LaTeX转换需求和复杂结构而更具挑战性。我们通过详细评估原始输出与对抗扰动输出之间的字符级和语义级变化，深入揭示模型的视觉解释与推理能力。该方法在ChatGPT上的应用进一步证明了其有效性，展现了其在真实场景中的实际意义。



## **41. A Benchmark for Ultra-High-Resolution Remote Sensing MLLMs**

超高分辨率遥感多模态大语言模型基准 cs.CV

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17319v1) [paper-pdf](https://arxiv.org/pdf/2512.17319v1)

**Confidence**: 0.85

**Authors**: Yunkai Dang, Meiyi Zhu, Donghao Wang, Yizhuo Zhang, Jiacheng Yang, Qi Fan, Yuekun Yang, Wenbin Li, Feng Miao, Yang Gao

**Abstract**: Multimodal large language models (MLLMs) demonstrate strong perception and reasoning performance on existing remote sensing (RS) benchmarks. However, most prior benchmarks rely on low-resolution imagery, and some high-resolution benchmarks suffer from flawed reasoning-task designs. We show that text-only LLMs can perform competitively with multimodal vision-language models on RS reasoning tasks without access to images, revealing a critical mismatch between current benchmarks and the intended evaluation of visual understanding. To enable faithful assessment, we introduce RSHR-Bench, a super-high-resolution benchmark for RS visual understanding and reasoning. RSHR-Bench contains 5,329 full-scene images with a long side of at least 4,000 pixels, with up to about 3 x 10^8 pixels per image, sourced from widely used RS corpora and UAV collections. We design four task families: multiple-choice VQA, open-ended VQA, image captioning, and single-image evaluation. These tasks cover nine perception categories and four reasoning types, supporting multi-turn and multi-image dialog. To reduce reliance on language priors, we apply adversarial filtering with strong LLMs followed by rigorous human verification. Overall, we construct 3,864 VQA tasks, 3,913 image captioning tasks, and 500 fully human-written or verified single-image evaluation VQA pairs. Evaluations across open-source, closed-source, and RS-specific VLMs reveal persistent performance gaps in super-high-resolution scenarios. Code: https://github.com/Yunkaidang/RSHR

摘要: 多模态大语言模型（MLLMs）在现有遥感（RS）基准测试中展现出强大的感知与推理能力。然而，多数先前基准依赖低分辨率影像，部分高分辨率基准存在推理任务设计缺陷。我们发现，纯文本大语言模型在无需访问图像的情况下，能在遥感推理任务中与多模态视觉语言模型竞争，这揭示了当前基准与视觉理解评估目标之间的严重错配。为实现可靠评估，我们提出了RSHR-Bench——一个面向遥感视觉理解与推理的超高分辨率基准。该基准包含5,329幅长边至少4,000像素的全景图像（单图最高约3×10^8像素），数据源自广泛使用的遥感数据集与无人机采集影像。我们设计了四类任务族：多项选择视觉问答、开放式视觉问答、图像描述生成及单图评估。这些任务涵盖九种感知类别与四种推理类型，支持多轮对话与多图像交互。为降低对语言先验的依赖，我们采用强语言模型进行对抗性筛选，并辅以严格人工验证。最终构建了3,864项视觉问答任务、3,913项图像描述任务，以及500组完全由人工撰写或验证的单图评估问答对。通过对开源、闭源及遥感专用视觉语言模型的评估，揭示了在超高分辨率场景下持续存在的性能差距。代码：https://github.com/Yunkaidang/RSHR



## **42. ToxiGAN: Toxic Data Augmentation via LLM-Guided Directional Adversarial Generation**

ToxiGAN：基于大语言模型引导的定向对抗生成的有害数据增强 cs.CL

This paper has been accepted to the main conference of EACL 2026

**SubmitDate**: 2026-01-06    [abs](http://arxiv.org/abs/2601.03121v1) [paper-pdf](https://arxiv.org/pdf/2601.03121v1)

**Confidence**: 0.85

**Authors**: Peiran Li, Jan Fillies, Adrian Paschke

**Abstract**: Augmenting toxic language data in a controllable and class-specific manner is crucial for improving robustness in toxicity classification, yet remains challenging due to limited supervision and distributional skew. We propose ToxiGAN, a class-aware text augmentation framework that combines adversarial generation with semantic guidance from large language models (LLMs). To address common issues in GAN-based augmentation such as mode collapse and semantic drift, ToxiGAN introduces a two-step directional training strategy and leverages LLM-generated neutral texts as semantic ballast. Unlike prior work that treats LLMs as static generators, our approach dynamically selects neutral exemplars to provide balanced guidance. Toxic samples are explicitly optimized to diverge from these exemplars, reinforcing class-specific contrastive signals. Experiments on four hate speech benchmarks show that ToxiGAN achieves the strongest average performance in both macro-F1 and hate-F1, consistently outperforming traditional and LLM-based augmentation methods. Ablation and sensitivity analyses further confirm the benefits of semantic ballast and directional training in enhancing classifier robustness.

摘要: 以可控且类别特定的方式增强有害语言数据对于提升毒性分类的鲁棒性至关重要，但由于监督有限和分布偏斜，这一任务仍具挑战性。我们提出ToxiGAN，一种结合对抗生成与大语言模型（LLM）语义引导的类别感知文本增强框架。为解决基于GAN的增强中常见的模式崩溃和语义漂移问题，ToxiGAN引入了两步定向训练策略，并利用LLM生成的中性文本作为语义压舱物。与先前将LLM视为静态生成器的工作不同，我们的方法动态选择中性示例以提供平衡引导。有害样本被显式优化以偏离这些示例，从而强化类别特定的对比信号。在四个仇恨言论基准测试上的实验表明，ToxiGAN在宏平均F1和仇恨F1指标上均取得最强平均性能，持续优于传统及基于LLM的增强方法。消融实验和敏感性分析进一步证实了语义压舱物和定向训练在提升分类器鲁棒性方面的优势。



## **43. MMBERT: Scaled Mixture-of-Experts Multimodal BERT for Robust Chinese Hate Speech Detection under Cloaking Perturbations**

MMBERT：基于可扩展专家混合的多模态BERT模型，用于抗伪装扰动的鲁棒性中文仇恨言论检测 cs.CL

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00760v1) [paper-pdf](https://arxiv.org/pdf/2508.00760v1)

**Confidence**: 0.85

**Authors**: Qiyao Xue, Yuchen Dou, Ryan Shi, Xiang Lorraine Li, Wei Gao

**Abstract**: Hate speech detection on Chinese social networks presents distinct challenges, particularly due to the widespread use of cloaking techniques designed to evade conventional text-based detection systems. Although large language models (LLMs) have recently improved hate speech detection capabilities, the majority of existing work has concentrated on English datasets, with limited attention given to multimodal strategies in the Chinese context. In this study, we propose MMBERT, a novel BERT-based multimodal framework that integrates textual, speech, and visual modalities through a Mixture-of-Experts (MoE) architecture. To address the instability associated with directly integrating MoE into BERT-based models, we develop a progressive three-stage training paradigm. MMBERT incorporates modality-specific experts, a shared self-attention mechanism, and a router-based expert allocation strategy to enhance robustness against adversarial perturbations. Empirical results in several Chinese hate speech datasets show that MMBERT significantly surpasses fine-tuned BERT-based encoder models, fine-tuned LLMs, and LLMs utilizing in-context learning approaches.

摘要: 中文社交网络中的仇恨言论检测面临独特挑战，尤其是由于广泛使用的伪装技术旨在规避传统的基于文本的检测系统。尽管大型语言模型（LLMs）近期提升了仇恨言论检测能力，但现有研究大多集中于英文数据集，对中文语境下的多模态策略关注有限。本研究提出MMBERT，一种基于BERT的新型多模态框架，通过专家混合（MoE）架构整合文本、语音和视觉模态。为解决直接将MoE集成到基于BERT的模型中的不稳定性问题，我们开发了一种渐进式三阶段训练范式。MMBERT包含模态特定专家、共享自注意力机制和基于路由器的专家分配策略，以增强对抗扰动的鲁棒性。在多个中文仇恨言论数据集上的实证结果表明，MMBERT显著优于微调的基于BERT的编码器模型、微调的LLMs以及采用上下文学习方法的LLMs。



## **44. The Coherence Trap: When MLLM-Crafted Narratives Exploit Manipulated Visual Contexts**

一致性陷阱：当MLLM构建的叙事利用被操纵的视觉语境 cs.CV

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2505.17476v2) [paper-pdf](https://arxiv.org/pdf/2505.17476v2)

**Confidence**: 0.85

**Authors**: Yuchen Zhang, Yaxiong Wang, Yujiao Wu, Lianwei Wu, Li Zhu, Zhedong Zheng

**Abstract**: The detection and grounding of multimedia manipulation has emerged as a critical challenge in combating AI-generated disinformation. While existing methods have made progress in recent years, we identify two fundamental limitations in current approaches: (1) Underestimation of MLLM-driven deception risk: prevailing techniques primarily address rule-based text manipulations, yet fail to account for sophisticated misinformation synthesized by multimodal large language models (MLLMs) that can dynamically generate semantically coherent, contextually plausible yet deceptive narratives conditioned on manipulated images; (2) Unrealistic misalignment artifacts: currently focused scenarios rely on artificially misaligned content that lacks semantic coherence, rendering them easily detectable. To address these gaps holistically, we propose a new adversarial pipeline that leverages MLLMs to generate high-risk disinformation. Our approach begins with constructing the MLLM-Driven Synthetic Multimodal (MDSM) dataset, where images are first altered using state-of-the-art editing techniques and then paired with MLLM-generated deceptive texts that maintain semantic consistency with the visual manipulations. Building upon this foundation, we present the Artifact-aware Manipulation Diagnosis via MLLM (AMD) framework featuring two key innovations: Artifact Pre-perception Encoding strategy and Manipulation-Oriented Reasoning, to tame MLLMs for the MDSM problem. Comprehensive experiments validate our framework's superior generalization capabilities as a unified architecture for detecting MLLM-powered multimodal deceptions. In cross-domain testing on the MDSM dataset, AMD achieves the best average performance, with 88.18 ACC, 60.25 mAP, and 61.02 mIoU scores.

摘要: 多媒体操纵的检测与溯源已成为对抗AI生成虚假信息的关键挑战。尽管现有方法近年来取得进展，但我们发现当前方法存在两个根本性局限：（1）低估MLLM驱动的欺骗风险：主流技术主要处理基于规则的文本操纵，却未能应对由多模态大语言模型（MLLMs）合成的复杂虚假信息——这些模型能基于被篡改图像动态生成语义连贯、语境合理却具有欺骗性的叙事；（2）不现实的错位伪影：当前聚焦的场景依赖缺乏语义连贯性的人工错位内容，使其易于被检测。为系统性解决这些缺陷，我们提出一种利用MLLMs生成高风险虚假信息的新型对抗流程。该方法首先构建MLLM驱动的合成多模态（MDSM）数据集，其中图像先通过最先进的编辑技术进行修改，再与MLLM生成的欺骗性文本配对，这些文本保持与视觉篡改的语义一致性。在此基础上，我们提出基于MLLM的伪影感知操纵诊断（AMD）框架，其具备两项关键创新：伪影预感知编码策略和面向操纵的推理机制，以驯化MLLMs解决MDSM问题。综合实验验证了我们框架作为检测MLLM驱动的多模态欺骗的统一架构具有卓越泛化能力。在MDSM数据集的跨域测试中，AMD取得最佳平均性能，达到88.18%准确率、60.25% mAP和61.02% mIoU分数。




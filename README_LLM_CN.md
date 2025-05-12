# Latest Large Language Model Attack Papers
**update at 2025-05-12 14:19:38**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AgentXploit: End-to-End Redteaming of Black-Box AI Agents**

AgentXploit：黑匣子人工智能代理的端到端红色团队 cs.CR

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2505.05849v1) [paper-pdf](http://arxiv.org/pdf/2505.05849v1)

**Authors**: Zhun Wang, Vincent Siu, Zhe Ye, Tianneng Shi, Yuzhou Nie, Xuandong Zhao, Chenguang Wang, Wenbo Guo, Dawn Song

**Abstract**: The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentXploit, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentXploit on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentXploit exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites.

摘要: 大型语言模型（LLM）强大的规划和推理能力促进了基于代理的系统的开发，这些系统能够利用外部工具并与日益复杂的环境进行交互。然而，这些强大的功能也引入了一个严重的安全风险：间接提示注入，这是一种复杂的攻击载体，通过操纵上下文信息而不是直接用户提示来损害这些代理的核心LLM。在这项工作中，我们提出了一个通用的黑匣子模糊框架AgentXploit，旨在自动发现和利用不同LLM代理之间的间接提示注入漏洞。我们的方法首先构建高质量的初始种子库，然后采用基于蒙特卡洛树搜索（MCTS）的种子选择算法来迭代细化输入，从而最大化发现代理弱点的可能性。我们在AgentDojo和VWA-adv这两个公共基准上评估了AgentXploit，它分别对基于o3-mini和GPT-4 o的代理实现了71%和70%的成功率，几乎是基线攻击性能的两倍。此外，AgentXploit在看不见的任务和内部LLM之间具有很强的可移植性，以及对抗防御的有希望的结果。除了基准评估之外，我们还将我们的攻击应用于现实环境中，成功地误导代理导航到任意URL，包括恶意网站。



## **2. Unified Attacks to Large Language Model Watermarks: Spoofing and Scrubbing in Unauthorized Knowledge Distillation**

对大型语言模型水印的统一攻击：未经授权的知识提炼中的欺骗和擦除 cs.CL

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2504.17480v3) [paper-pdf](http://arxiv.org/pdf/2504.17480v3)

**Authors**: Xin Yi, Yue Li, Shunfan Zheng, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable.

摘要: 水印已成为打击错误信息和保护大型语言模型（LLM）知识产权的关键技术。最近的一项发现称为水印放射性，揭示了教师模型中嵌入的水印可以通过知识蒸馏被学生模型继承。从积极的方面来说，这种继承允许通过识别学生模型中的水印痕迹来检测未经授权的知识提炼。然而，水印对擦洗攻击的鲁棒性及其在未经授权的知识提炼下面对欺骗攻击时的不可伪造性在很大程度上仍然没有被探索。现有的水印攻击方法要么假设访问模型内部，要么不能同时支持擦洗和欺骗攻击。在这项工作中，我们提出了对比解码引导的知识蒸馏（CDG-KD），一个统一的框架，使未经授权的知识蒸馏下的双向攻击。我们的方法采用对比解码提取损坏或放大的水印文本，通过比较输出的学生模型和弱水印的参考，然后通过双向蒸馏训练新的学生模型能够水印去除和水印伪造，分别。大量的实验表明，CDG-KD有效地执行攻击，同时保持蒸馏模型的一般性能。我们的研究结果强调了开发稳健且不可伪造的水印方案的迫切需要。



## **3. LiteLMGuard: Seamless and Lightweight On-Device Prompt Filtering for Safeguarding Small Language Models against Quantization-induced Risks and Vulnerabilities**

LiteLMGGuard：无缝且轻量级的设备上提示过滤，用于保护小语言模型免受量化引发的风险和漏洞的影响 cs.CR

14 pages, 18 figures, and 4 tables

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05619v1) [paper-pdf](http://arxiv.org/pdf/2505.05619v1)

**Authors**: Kalyan Nakka, Jimmy Dani, Ausmit Mondal, Nitesh Saxena

**Abstract**: The growing adoption of Large Language Models (LLMs) has influenced the development of their lighter counterparts-Small Language Models (SLMs)-to enable on-device deployment across smartphones and edge devices. These SLMs offer enhanced privacy, reduced latency, server-free functionality, and improved user experience. However, due to resource constraints of on-device environment, SLMs undergo size optimization through compression techniques like quantization, which can inadvertently introduce fairness, ethical and privacy risks. Critically, quantized SLMs may respond to harmful queries directly, without requiring adversarial manipulation, raising significant safety and trust concerns.   To address this, we propose LiteLMGuard (LLMG), an on-device prompt guard that provides real-time, prompt-level defense for quantized SLMs. Additionally, our prompt guard is designed to be model-agnostic such that it can be seamlessly integrated with any SLM, operating independently of underlying architectures. Our LLMG formalizes prompt filtering as a deep learning (DL)-based prompt answerability classification task, leveraging semantic understanding to determine whether a query should be answered by any SLM. Using our curated dataset, Answerable-or-Not, we trained and fine-tuned several DL models and selected ELECTRA as the candidate, with 97.75% answerability classification accuracy.   Our safety effectiveness evaluations demonstrate that LLMG defends against over 87% of harmful prompts, including both direct instruction and jailbreak attack strategies. We further showcase its ability to mitigate the Open Knowledge Attacks, where compromised SLMs provide unsafe responses without adversarial prompting. In terms of prompt filtering effectiveness, LLMG achieves near state-of-the-art filtering accuracy of 94%, with an average latency of 135 ms, incurring negligible overhead for users.

摘要: 大型语言模型（LLM）的日益采用影响了其更轻的同类产品--小型语言模型（SLM）--的发展，以实现跨智能手机和边缘设备的设备上部署。这些STM提供增强的隐私、减少的延迟、无服务器功能和改善的用户体验。然而，由于设备上环境的资源限制，STM通过量化等压缩技术进行尺寸优化，这可能会无意中引入公平性、道德和隐私风险。至关重要的是，量化的SLC可以直接响应有害查询，而不需要对抗性操纵，从而引发重大的安全和信任问题。   为了解决这个问题，我们提出了LiteLMGard（LLMG），这是一种设备上提示保护，为量化的STM提供实时、预算级防御。此外，我们的提示卫士设计为模型不可知，因此它可以与任何SPL无缝集成，独立于底层架构运行。我们的LLMG将提示过滤形式化为基于深度学习（DL）的提示可回答性分类任务，利用语义理解来确定查询是否应该由任何SPL回答。使用我们精心策划的数据集“可供选择”，我们训练和微调了几个DL模型，并选择ELECTRA作为候选模型，其回答性分类准确率为97.75%。   我们的安全有效性评估表明，LLMG可以抵御超过87%的有害提示，包括直接指令和越狱攻击策略。我们进一步展示了其缓解开放知识攻击的能力，其中受攻击的STM在没有对抗提示的情况下提供不安全的响应。在即时过滤有效性方面，LLMG实现了94%的接近最先进的过滤准确率，平均延迟为135 ms，为用户带来的负担可以忽略不计。



## **4. Towards the Worst-case Robustness of Large Language Models**

走向大型语言模型的最坏情况稳健性 cs.LG

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2501.19040v2) [paper-pdf](http://arxiv.org/pdf/2501.19040v2)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of large language models to adversarial attacks, where adversaries craft specific input sequences to induce harmful, violent, private, or incorrect outputs. In this work, we study their worst-case robustness, i.e., whether an adversarial example exists that leads to such undesirable outputs. We upper bound the worst-case robustness using stronger white-box attacks, indicating that most current deterministic defenses achieve nearly 0\% worst-case robustness. We propose a general tight lower bound for randomized smoothing using fractional knapsack solvers or 0-1 knapsack solvers, and using them to bound the worst-case robustness of all stochastic defenses. Based on these solvers, we provide theoretical lower bounds for several previous empirical defenses. For example, we certify the robustness of a specific case, smoothing using a uniform kernel, against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.

摘要: 最近的研究揭示了大型语言模型容易受到对抗攻击，对手会精心设计特定的输入序列来引发有害、暴力、私密或错误的输出。在这项工作中，我们研究了它们的最坏情况稳健性，即是否存在导致此类不良结果的对抗性例子。我们使用更强的白盒攻击来对最坏情况的稳健性进行上限，这表明当前大多数确定性防御实现了近0%的最坏情况的稳健性。我们提出了使用分数背包求解器或0-1背包求解器的随机平滑的一般紧下界，并使用它们来限制所有随机防御的最坏情况稳健性。基于这些求解器，我们为之前的几个经验防御提供了理论下限。例如，我们证明了特定情况的稳健性，使用统一核进行平滑，针对\texttit {任何可能的攻击}，平均$\ell_0 $扰动为2.02或平均后缀长度为6.41。



## **5. Jailbreaking and Mitigation of Vulnerabilities in Large Language Models**

大型语言模型中的漏洞越狱和缓解 cs.CR

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2410.15236v2) [paper-pdf](http://arxiv.org/pdf/2410.15236v2)

**Authors**: Benji Peng, Keyu Chen, Qian Niu, Ziqian Bi, Ming Liu, Pohsun Feng, Tianyang Wang, Lawrence K. Q. Yan, Yizhu Wen, Yichao Zhang, Caitlyn Heqi Yin

**Abstract**: Large Language Models (LLMs) have transformed artificial intelligence by advancing natural language understanding and generation, enabling applications across fields beyond healthcare, software engineering, and conversational systems. Despite these advancements in the past few years, LLMs have shown considerable vulnerabilities, particularly to prompt injection and jailbreaking attacks. This review analyzes the state of research on these vulnerabilities and presents available defense strategies. We roughly categorize attack approaches into prompt-based, model-based, multimodal, and multilingual, covering techniques such as adversarial prompting, backdoor injections, and cross-modality exploits. We also review various defense mechanisms, including prompt filtering, transformation, alignment techniques, multi-agent defenses, and self-regulation, evaluating their strengths and shortcomings. We also discuss key metrics and benchmarks used to assess LLM safety and robustness, noting challenges like the quantification of attack success in interactive contexts and biases in existing datasets. Identifying current research gaps, we suggest future directions for resilient alignment strategies, advanced defenses against evolving attacks, automation of jailbreak detection, and consideration of ethical and societal impacts. This review emphasizes the need for continued research and cooperation within the AI community to enhance LLM security and ensure their safe deployment.

摘要: 大型语言模型（LLM）通过推进自然语言理解和生成，改变了人工智能，实现了医疗保健、软件工程和会话系统以外的应用。尽管过去几年取得了这些进步，但LLM仍表现出相当大的漏洞，特别是在引发注射和越狱攻击方面。本评论分析了这些漏洞的研究状况，并提出了可用的防御策略。我们大致将攻击方法分为基于模型的，基于模型的，多模式的和多语言的，涵盖了对抗性提示，后门注入和跨模式利用等技术。我们还回顾了各种防御机制，包括即时过滤、转换、对齐技术、多智能体防御和自我调节，评估它们的优点和缺点。我们还讨论了用于评估LLM安全性和稳健性的关键指标和基准，并指出了交互式环境中攻击成功的量化以及现有数据集中的偏差等挑战。通过识别当前的研究差距，我们提出了弹性对齐策略、针对不断发展的攻击的先进防御、越狱检测自动化以及道德和社会影响的未来方向。该审查强调了人工智能社区内持续研究与合作的必要性，以增强LLM安全性并确保其安全部署。



## **6. Stealthy LLM-Driven Data Poisoning Attacks Against Embedding-Based Retrieval-Augmented Recommender Systems**

针对基于嵌入的检索增强推荐系统的隐形LLM驱动的数据中毒攻击 cs.IR

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05196v1) [paper-pdf](http://arxiv.org/pdf/2505.05196v1)

**Authors**: Fatemeh Nazary, Yashar Deldjoo, Tommaso Di Noia, Eugenio Di Sciascio

**Abstract**: We present a systematic study of provider-side data poisoning in retrieval-augmented recommender systems (RAG-based). By modifying only a small fraction of tokens within item descriptions -- for instance, adding emotional keywords or borrowing phrases from semantically related items -- an attacker can significantly promote or demote targeted items. We formalize these attacks under token-edit and semantic-similarity constraints, and we examine their effectiveness in both promotion (long-tail items) and demotion (short-head items) scenarios. Our experiments on MovieLens, using two large language model (LLM) retrieval modules, show that even subtle attacks shift final rankings and item exposures while eluding naive detection. The results underscore the vulnerability of RAG-based pipelines to small-scale metadata rewrites and emphasize the need for robust textual consistency checks and provenance tracking to thwart stealthy provider-side poisoning.

摘要: 我们对检索增强推荐系统（基于RAG）中的提供商端数据中毒进行了系统研究。通过仅修改物品描述中的一小部分标记--例如添加情感关键词或借用语义相关物品的短语--攻击者可以显着提升或降级目标物品。我们在标记编辑和语义相似性约束下对这些攻击进行形式化，并检查它们在晋升（长尾项）和降级（短头项）场景中的有效性。我们使用两个大型语言模型（LLM）检索模块在MovieLens上进行的实验表明，即使是微妙的攻击也会改变最终排名和项目暴露，同时避免天真的检测。结果强调了基于RAG的管道对小规模元数据重写的脆弱性，并强调需要强大的文本一致性检查和出处跟踪来阻止隐形的提供商端中毒。



## **7. Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks**

通过自信息重写攻击揭示文本水印的弱点 cs.LG

ICML 2025 Accpeted

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05190v1) [paper-pdf](http://arxiv.org/pdf/2505.05190v1)

**Authors**: Yixin Cheng, Hongcheng Guo, Yangming Li, Leonid Sigal

**Abstract**: Text watermarking aims to subtly embed statistical signals into text by controlling the Large Language Model (LLM)'s sampling process, enabling watermark detectors to verify that the output was generated by the specified model. The robustness of these watermarking algorithms has become a key factor in evaluating their effectiveness. Current text watermarking algorithms embed watermarks in high-entropy tokens to ensure text quality. In this paper, we reveal that this seemingly benign design can be exploited by attackers, posing a significant risk to the robustness of the watermark. We introduce a generic efficient paraphrasing attack, the Self-Information Rewrite Attack (SIRA), which leverages the vulnerability by calculating the self-information of each token to identify potential pattern tokens and perform targeted attack. Our work exposes a widely prevalent vulnerability in current watermarking algorithms. The experimental results show SIRA achieves nearly 100% attack success rates on seven recent watermarking methods with only 0.88 USD per million tokens cost. Our approach does not require any access to the watermark algorithms or the watermarked LLM and can seamlessly transfer to any LLM as the attack model, even mobile-level models. Our findings highlight the urgent need for more robust watermarking.

摘要: 文本水印旨在通过控制大型语言模型（LLM）的采样过程将统计信号巧妙地嵌入到文本中，使水印检测器能够验证输出是否由指定模型生成。这些水印算法的鲁棒性已成为评估其有效性的关键因素。当前的文本水印算法将水印嵌入高熵令牌中以确保文本质量。在本文中，我们揭示了这种看似良性的设计可能会被攻击者利用，从而对水印的稳健性构成重大风险。我们引入了一种通用的高效解释攻击，即自我信息重写攻击（SIRA），它通过计算每个令牌的自我信息来利用漏洞来识别潜在的模式令牌并执行有针对性的攻击。我们的工作揭示了当前水印算法中广泛存在的漏洞。实验结果表明，SIRA对最近的七种水印方法的攻击成功率接近100%，每百万个代币的成本仅为0.88美元。我们的方法不需要对水印算法或带水印的LLM进行任何访问，并且可以无缝地转移到任何LLM作为攻击模型，甚至是移动级模型。我们的研究结果凸显了对更鲁棒的水印的迫切需求。



## **8. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

X-Transfer攻击：CLIP上的超级可转移对抗攻击 cs.CV

ICML 2025

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05528v1) [paper-pdf](http://arxiv.org/pdf/2505.05528v1)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.

摘要: 随着对比图像预训练（CLIP）模型越来越多地被用于各种下游任务并集成到大型视觉语言模型（VLM）中，它们对对抗性扰动的敏感性已成为一个关键问题。在这项工作中，我们介绍了\textbf {X-Transfer}，一种新的攻击方法，暴露了CLIP中的一个普遍的对抗性漏洞。X-Transfer生成一个通用对抗扰动（Universal Adversarial Perturbation，UAP），能够欺骗不同样本、任务和域中的各种CLIP编码器和下游VLM。我们将此属性称为\textbf {super transferability}--一个同时实现跨数据、跨域、跨模型和跨任务对抗性可转移性的单一扰动。这是通过\textBF {代理缩放}来实现的，这是我们方法的一个关键创新。与依赖于固定代理模型（扩展计算密集型）的现有方法不同，X-Transfer采用高效的代理扩展策略，可以从大搜索空间中动态选择合适代理的一小子集。广泛的评估表明，X-Transfer的性能显着优于之前最先进的UAP方法，为跨CLIP模型的对抗性可移植性建立了新的基准。该代码可在我们的\href {https：//github.com/HanxunH/XTransferBench}{GitHub存储库}中公开获取。



## **9. Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction**

可靠地限制假阳性：通过多尺度保形预测的零镜头机器生成文本检测框架 cs.CL

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05084v1) [paper-pdf](http://arxiv.org/pdf/2505.05084v1)

**Authors**: Xiaowei Zhu, Yubing Ren, Yanan Cao, Xixun Lin, Fang Fang, Yangxi Li

**Abstract**: The rapid advancement of large language models has raised significant concerns regarding their potential misuse by malicious actors. As a result, developing effective detectors to mitigate these risks has become a critical priority. However, most existing detection methods focus excessively on detection accuracy, often neglecting the societal risks posed by high false positive rates (FPRs). This paper addresses this issue by leveraging Conformal Prediction (CP), which effectively constrains the upper bound of FPRs. While directly applying CP constrains FPRs, it also leads to a significant reduction in detection performance. To overcome this trade-off, this paper proposes a Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction (MCP), which both enforces the FPR constraint and improves detection performance. This paper also introduces RealDet, a high-quality dataset that spans a wide range of domains, ensuring realistic calibration and enabling superior detection performance when combined with MCP. Empirical evaluations demonstrate that MCP effectively constrains FPRs, significantly enhances detection performance, and increases robustness against adversarial attacks across multiple detectors and datasets.

摘要: 大型语言模型的快速发展引发了人们对其潜在被恶意行为者滥用的严重担忧。因此，开发有效的探测器来减轻这些风险已成为当务之急。然而，大多数现有的检测方法过度关注检测准确性，往往忽视了高假阳性率（FPR）带来的社会风险。本文通过利用保形预测（CP）来解决这个问题，该预测有效地限制了FPR的上界。虽然直接应用CP约束FPR，但也会导致检测性能显着降低。为了克服这种权衡，本文提出了一种通过多尺度保形预测（LCP）的零镜头机器生成文本检测框架，该框架既强制执行FPR约束又提高检测性能。本文还介绍了RealDet，这是一个跨越广泛领域的高质量数据集，可确保真实的校准并在与HCP结合时实现卓越的检测性能。经验评估表明，LCP有效地约束了FPR，显着增强了检测性能，并增强了针对多个检测器和数据集的对抗攻击的鲁棒性。



## **10. Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs**

Red联手机器思维：LLM中即时注射和越狱漏洞的系统评估 cs.CR

7 Pages, 6 Figures

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04806v1) [paper-pdf](http://arxiv.org/pdf/2505.04806v1)

**Authors**: Chetan Pathade

**Abstract**: Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.

摘要: 大型语言模型（LLM）越来越多地集成到消费者和企业应用程序中。尽管它们有能力，但它们仍然容易受到对抗攻击，例如超越对齐保障措施的立即注射和越狱。本文对针对各种最先进的法学硕士的越狱策略进行了系统调查。我们对1，400多个对抗提示进行了分类，分析了它们对GPT-4、Claude 2、Mistral 7 B和Vicuna的成功，并检查它们的概括性和构造逻辑。我们进一步提出分层缓解策略，并推荐混合红色团队和沙箱方法以实现强大的LLM安全性。



## **11. Safeguard-by-Development: A Privacy-Enhanced Development Paradigm for Multi-Agent Collaboration Systems**

开发保障：多代理协作系统的隐私增强开发范式 cs.CR

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04799v1) [paper-pdf](http://arxiv.org/pdf/2505.04799v1)

**Authors**: Jian Cui, Zichuan Li, Luyi Xing, Xiaojing Liao

**Abstract**: Multi-agent collaboration systems (MACS), powered by large language models (LLMs), solve complex problems efficiently by leveraging each agent's specialization and communication between agents. However, the inherent exchange of information between agents and their interaction with external environments, such as LLM, tools, and users, inevitably introduces significant risks of sensitive data leakage, including vulnerabilities to attacks like prompt injection and reconnaissance. Existing MACS fail to enable privacy controls, making it challenging to manage sensitive information securely. In this paper, we take the first step to address the MACS's data leakage threat at the system development level through a privacy-enhanced development paradigm, Maris. Maris enables rigorous message flow control within MACS by embedding reference monitors into key multi-agent conversation components. We implemented Maris as an integral part of AutoGen, a widely adopted open-source multi-agent development framework. Then, we evaluate Maris for its effectiveness and performance overhead on privacy-critical MACS use cases, including healthcare, supply chain optimization, and personalized recommendation system. The result shows that Maris achieves satisfactory effectiveness, performance overhead and practicability for adoption.

摘要: 多代理协作系统（MACS）由大型语言模型（LLM）提供支持，通过利用每个代理的专业化和代理之间的通信来有效地解决复杂问题。然而，代理之间固有的信息交换及其与外部环境（例如LLM、工具和用户）的交互，不可避免地会带来敏感数据泄露的重大风险，包括即时注入和侦察等攻击的漏洞。现有的MACS无法启用隐私控制，因此安全地管理敏感信息具有挑战性。在本文中，我们迈出了第一步，通过隐私增强的开发范式Maris在系统开发层面解决MACS的数据泄露威胁。Maris通过将引用监视器嵌入到关键的多代理对话组件中来在MACS内实现严格的消息流控制。我们将Maris实施为AutoGen的一部分，AutoGen是一个广泛采用的开源多代理开发框架。然后，我们评估Maris在隐私关键MACS用例（包括医疗保健、供应链优化和个性化推荐系统）上的有效性和性能费用。结果表明，Maris达到了令人满意的有效性、性能负担和采用的实用性。



## **12. A Proposal for Evaluating the Operational Risk for ChatBots based on Large Language Models**

基于大型语言模型评估ChatBots运营风险的提案 cs.CR

21 pages

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04784v1) [paper-pdf](http://arxiv.org/pdf/2505.04784v1)

**Authors**: Pedro Pinacho-Davidson, Fernando Gutierrez, Pablo Zapata, Rodolfo Vergara, Pablo Aqueveque

**Abstract**: The emergence of Generative AI (Gen AI) and Large Language Models (LLMs) has enabled more advanced chatbots capable of human-like interactions. However, these conversational agents introduce a broader set of operational risks that extend beyond traditional cybersecurity considerations. In this work, we propose a novel, instrumented risk-assessment metric that simultaneously evaluates potential threats to three key stakeholders: the service-providing organization, end users, and third parties. Our approach incorporates the technical complexity required to induce erroneous behaviors in the chatbot--ranging from non-induced failures to advanced prompt-injection attacks--as well as contextual factors such as the target industry, user age range, and vulnerability severity. To validate our metric, we leverage Garak, an open-source framework for LLM vulnerability testing. We further enhance Garak to capture a variety of threat vectors (e.g., misinformation, code hallucinations, social engineering, and malicious code generation). Our methodology is demonstrated in a scenario involving chatbots that employ retrieval-augmented generation (RAG), showing how the aggregated risk scores guide both short-term mitigation and longer-term improvements in model design and deployment. The results underscore the importance of multi-dimensional risk assessments in operationalizing secure, reliable AI-driven conversational systems.

摘要: 生成式人工智能（Gen AI）和大型语言模型（LLM）的出现使更先进的聊天机器人能够进行类人交互。然而，这些对话代理引入了一系列更广泛的运营风险，超出了传统的网络安全考虑。在这项工作中，我们提出了一种新颖的、工具化的风险评估指标，该指标同时评估对三个关键利益相关者的潜在威胁：服务提供组织、最终用户和第三方。我们的方法结合了在聊天机器人中诱导错误行为所需的技术复杂性（从非诱导故障到高级预算注入攻击），以及目标行业、用户年龄范围和漏洞严重性等上下文因素。为了验证我们的指标，我们利用Garak，这是一个LLM漏洞测试的开源框架。我们进一步增强Garak以捕获各种威胁载体（例如，错误信息、代码幻觉、社会工程和恶意代码生成）。我们的方法在涉及采用检索增强生成（RAG）的聊天机器人的场景中进行了演示，展示了汇总风险评分如何指导模型设计和部署的短期缓解和长期改进。结果强调了多维风险评估在运营安全、可靠的人工智能驱动对话系统方面的重要性。



## **13. ACE: A Security Architecture for LLM-Integrated App Systems**

ACE：LLM集成应用程序系统的安全架构 cs.CR

21 pages, 13 figures; clarify relation to indirect prompt injection  attacks

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2504.20984v2) [paper-pdf](http://arxiv.org/pdf/2504.20984v2)

**Authors**: Evan Li, Tushin Mallick, Evan Rose, William Robertson, Alina Oprea, Cristina Nita-Rotaru

**Abstract**: LLM-integrated app systems extend the utility of Large Language Models (LLMs) with third-party apps that are invoked by a system LLM using interleaved planning and execution phases to answer user queries. These systems introduce new attack vectors where malicious apps can cause integrity violation of planning or execution, availability breakdown, or privacy compromise during execution.   In this work, we identify new attacks impacting the integrity of planning, as well as the integrity and availability of execution in LLM-integrated apps, and demonstrate them against IsolateGPT, a recent solution designed to mitigate attacks from malicious apps. We propose Abstract-Concrete-Execute (ACE), a new secure architecture for LLM-integrated app systems that provides security guarantees for system planning and execution. Specifically, ACE decouples planning into two phases by first creating an abstract execution plan using only trusted information, and then mapping the abstract plan to a concrete plan using installed system apps. We verify that the plans generated by our system satisfy user-specified secure information flow constraints via static analysis on the structured plan output. During execution, ACE enforces data and capability barriers between apps, and ensures that the execution is conducted according to the trusted abstract plan. We show experimentally that our system is secure against attacks from the INJECAGENT benchmark, a standard benchmark for control flow integrity in the face of indirect prompt injection attacks, and our newly introduced attacks. Our architecture represents a significant advancement towards hardening LLM-based systems containing system facilities of varying levels of trustworthiness.

摘要: LLM集成的应用程序系统通过第三方应用程序扩展了大型语言模型（LLM）的实用性，第三方应用程序由系统LLM使用交错的规划和执行阶段调用，以回答用户查询。这些系统引入了新的攻击载体，恶意应用程序可能会导致规划或执行的完整性违反、可用性崩溃或执行期间的隐私受到损害。   在这项工作中，我们识别了影响规划完整性以及LLM集成应用程序中执行完整性和可用性的新攻击，并针对IsolateGPT（旨在减轻恶意应用程序攻击的最新解决方案）进行演示。我们提出Abstract-Concrete-Execute（ACE），这是一种针对LLM集成应用程序系统的新安全架构，为系统规划和执行提供安全保障。具体来说，ACE将规划分为两个阶段，首先仅使用可信信息创建抽象执行计划，然后使用已安装的系统应用程序将抽象计划映射到具体计划。我们通过对结构化计划输出的静态分析来验证系统生成的计划是否满足用户指定的安全信息流约束。在执行过程中，ACE在应用程序之间强制设置数据和能力障碍，并确保执行按照可信的抽象计划进行。我们通过实验证明，我们的系统可以抵御来自INJECAGENT基准测试（面对间接提示注入攻击时控制流完整性的标准基准）的攻击，以及我们新引入的攻击。我们的架构代表了强化基于LLM的系统的重大进步，该系统包含不同可信度级别的系统设施。



## **14. Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization**

以毒攻毒：通过奖励中和防御恶意RL微调 cs.LG

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04578v1) [paper-pdf](http://arxiv.org/pdf/2505.04578v1)

**Authors**: Wenjun Cao

**Abstract**: Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models.

摘要: 强化学习（RL）微调改变了大型语言模型，同时创建了我们实验验证的漏洞：我们的实验表明，恶意RL微调以显着的效率突破了安全护栏，只需要50个步骤和最少的对抗提示，有害的升级从0-2升级到7-9。这种攻击载体特别威胁具有参数级访问权限的开源模型。事实证明，针对监督式微调的现有防御措施对RL的动态反馈机制无效。我们引入了奖励中和，这是第一个专门针对RL微调攻击而设计的防御框架，建立了简洁的拒绝模式，使恶意奖励信号无效。我们的方法训练模型以产生攻击者无法利用的最小信息拒绝，系统性地抵消针对有害输出进行优化的尝试。实验验证了我们的方法在200次攻击步骤后保持较低的有害分数（不大于2），而标准模型迅速恶化。这项工作提供了第一个建设性的证据，证明可以实现针对日益容易获得的RL攻击的强大防御，解决了开权模型的关键安全差距。



## **15. An LLM-based Self-Evolving Security Framework for 6G Space-Air-Ground Integrated Networks**

基于LLM的6G空地综合网络自进化安全框架 cs.CR

Accepted by IEEE Communications Magazine

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.03161v2) [paper-pdf](http://arxiv.org/pdf/2505.03161v2)

**Authors**: Qi Qin, Xinye Cao, Guoshun Nan, Sihan Chen, Rushan Li, Li Su, Haitao Du, Qimei Cui, Pengxuan Mao, Xiaofeng Tao, Tony Q. S. Quek

**Abstract**: Recently emerged 6G space-air-ground integrated networks (SAGINs), which integrate satellites, aerial networks, and terrestrial communications, offer ubiquitous coverage for various mobile applications. However, the highly dynamic, open, and heterogeneous nature of SAGINs poses severe security issues. Forming a defense line of SAGINs suffers from two preliminary challenges: 1) accurately understanding massive unstructured multi-dimensional threat information to generate defense strategies against various malicious attacks, 2) rapidly adapting to potential unknown threats to yield more effective security strategies. To tackle the above two challenges, we propose a novel security framework for SAGINs based on Large Language Models (LLMs), which consists of two key ingredients LLM-6GNG and 6G-INST. Our proposed LLM-6GNG leverages refined chain-of-thought (CoT) reasoning and dynamic multi-agent mechanisms to analyze massive unstructured multi-dimensional threat data and generate comprehensive security strategies, thus addressing the first challenge. Our proposed 6G-INST relies on a novel self-evolving method to automatically update LLM-6GNG, enabling it to accommodate unknown threats under dynamic communication environments, thereby addressing the second challenge. Additionally, we prototype the proposed framework with ns-3, OpenAirInterface (OAI), and software-defined radio (SDR). Experiments on three benchmarks demonstrate the effectiveness of our framework. The results show that our framework produces highly accurate security strategies that remain robust against a variety of unknown attacks. We will release our code to contribute to the community.

摘要: 最近出现的6 G空-空-地综合网络（SAGER）集成了卫星、空中网络和地面通信，为各种移动应用提供无处不在的覆盖。然而，SATIN的高度动态、开放和异类性质带来了严重的安全问题。形成SATIN防线面临两个初步挑战：1）准确理解大量非结构化多维威胁信息，以生成针对各种恶意攻击的防御策略，2）快速适应潜在的未知威胁，以生成更有效的安全策略。为了应对上述两个挑战，我们提出了一种基于大型语言模型（LLM）的SAGER的新型安全框架，该框架由LLM-6 GNG和6 G-INST两个关键成分组成。我们提出的LLM-6 GNG利用精细化思想链（CoT）推理和动态多代理机制来分析大量非结构化多维威胁数据并生成全面的安全策略，从而解决第一个挑战。我们提出的6 G-INST依赖于一种新颖的自我进化方法来自动更新LLM-6 GNG，使其能够适应动态通信环境下的未知威胁，从而解决第二个挑战。此外，我们还使用ns-3、OpenAir接口（OAI）和软件定义无线电（SDR）对拟议框架进行了原型化。三个基准测试的实验证明了我们框架的有效性。结果表明，我们的框架可以生成高度准确的安全策略，并且能够抵御各种未知攻击。我们将发布我们的代码为社区做出贡献。



## **16. OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models**

Obliviate：针对大型语言模型的稳健且实用的机器去学习 cs.CL

18 pages, 2 figures

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04416v1) [paper-pdf](http://arxiv.org/pdf/2505.04416v1)

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose OBLIVIATE, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA), it ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including the Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: forget quality (new document-level memorization score), model utility, and fluency. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios.

摘要: 在广泛的库中训练的大型语言模型（LLM）存在记忆敏感、受版权保护或有毒内容的风险。为了解决这个问题，我们提出了OBLIATE，这是一个强大的去学习框架，可以删除目标数据，同时保留模型效用。该框架遵循一个结构化过程：提取目标令牌、构建保留集以及使用定制的损失函数进行微调，该函数包括三个部分--掩蔽、蒸馏和世界事实。使用低级适配器（LoRA），它可以在不影响取消学习质量的情况下确保效率。我们使用一套全面的指标对多个数据集进行实验，包括哈利·波特系列、WMDP和TOFU，包括忘记质量（新的文档级记忆分数）、模型效用和流利度。结果证明了它在抵抗隶属度推理攻击、最大限度地减少对保留数据的影响以及在不同场景下保持稳健性方面的有效性。



## **17. The Aloe Family Recipe for Open and Specialized Healthcare LLMs**

面向开放和专业医疗LL的Aloe家族食谱 cs.CL

arXiv admin note: substantial text overlap with arXiv:2405.01886

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04388v1) [paper-pdf](http://arxiv.org/pdf/2505.04388v1)

**Authors**: Dario Garcia-Gasulla, Jordi Bayarri-Planas, Ashwin Kumar Gururajan, Enrique Lopez-Cuena, Adrian Tormos, Daniel Hinjos, Pablo Bernabeu-Perez, Anna Arias-Duart, Pablo Agustin Martin-Torres, Marta Gonzalez-Mallo, Sergio Alvarez-Napagao, Eduard Ayguadé-Parra, Ulises Cortés

**Abstract**: Purpose: With advancements in Large Language Models (LLMs) for healthcare, the need arises for competitive open-source models to protect the public interest. This work contributes to the field of open medical LLMs by optimizing key stages of data preprocessing and training, while showing how to improve model safety (through DPO) and efficacy (through RAG). The evaluation methodology used, which includes four different types of tests, defines a new standard for the field. The resultant models, shown to be competitive with the best private alternatives, are released with a permisive license.   Methods: Building on top of strong base models like Llama 3.1 and Qwen 2.5, Aloe Beta uses a custom dataset to enhance public data with synthetic Chain of Thought examples. The models undergo alignment with Direct Preference Optimization, emphasizing ethical and policy-aligned performance in the presence of jailbreaking attacks. Evaluation includes close-ended, open-ended, safety and human assessments, to maximize the reliability of results.   Results: Recommendations are made across the entire pipeline, backed by the solid performance of the Aloe Family. These models deliver competitive performance across healthcare benchmarks and medical fields, and are often preferred by healthcare professionals. On bias and toxicity, the Aloe Beta models significantly improve safety, showing resilience to unseen jailbreaking attacks. For a responsible release, a detailed risk assessment specific to healthcare is attached to the Aloe Family models.   Conclusion: The Aloe Beta models, and the recipe that leads to them, are a significant contribution to the open-source medical LLM field, offering top-of-the-line performance while maintaining high ethical requirements. This work sets a new standard for developing and reporting aligned LLMs in healthcare.

摘要: 目的：随着医疗保健大型语言模型（LLM）的进步，需要有竞争力的开源模型来保护公共利益。这项工作通过优化数据预处理和训练的关键阶段，同时展示如何提高模型安全性（通过DPO）和有效性（通过RAG），为开放医学LLM领域做出了贡献。所使用的评估方法包括四种不同类型的测试，为该领域定义了新的标准。由此产生的模型被证明与最好的私人替代品具有竞争力，并且在许可证下发布。   方法：Aloe Beta建立在Llama 3.1和Qwen 2.5等强大基础模型的基础上，使用自定义数据集通过合成的思想链示例增强公共数据。这些模型与直接偏好优化保持一致，强调在存在越狱攻击时的道德和政策一致的性能。评估包括封闭式、开放式、安全性和人为评估，以最大限度地提高结果的可靠性。   结果：在Aloe系列的稳健表现的支持下，整个管道都提出了建议。这些模型在医疗保健基准和医疗领域提供有竞争力的性能，并且通常受到医疗保健专业人士的青睐。在偏见和毒性方面，Aloe Beta模型显着提高了安全性，表现出对不可见越狱攻击的韧性。为了实现负责任的发布，Aloe Family模型附带了针对医疗保健的详细风险评估。   结论：Aloe Beta模型及其配方是对开源医学LLM领域的重大贡献，在保持高道德要求的同时提供顶级性能。这项工作为开发和报告医疗保健领域一致的LLM设定了新标准。



## **18. REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLM**

REVEAL：Vision LLM图像输入危害的多回合评估 cs.CL

13 pages (8 main), to be published in IJCAI 2025

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04673v1) [paper-pdf](http://arxiv.org/pdf/2505.04673v1)

**Authors**: Madhur Jindal, Saurabh Deshpande

**Abstract**: Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o.   We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate ($16.55 \%$) while Qwen2-VL showed the highest MT refusal rate ($19.1 \%$).

摘要: Vision大型语言模型（VLLM）将图像处理能力与文本理解集成，从而增强用户交互并扩展应用程序领域，代表了人工智能的重大进步。然而，它们日益增加的复杂性带来了新的安全和道德挑战，特别是在多模式和多回合对话中。传统的安全评估框架专为基于文本的单轮交互而设计，不足以解决这些复杂性。为了弥合这一差距，我们引入了REVEAL（视觉启用的AI LLM负责任评估）框架，这是一个可扩展和自动化的管道，用于评估VLLM中的图像输入伤害。REVEAL包括自动图像挖掘、合成对抗数据生成、使用渐强攻击策略的多轮对话扩展，以及通过GPT-4 o等评估器进行的全面危害评估。   我们广泛评估了五种最先进的VLLM：GPT-4 o、Llama-3.2、Qwen 2-BL、Phi3.5V和Pixtral，涵盖三个重要的伤害类别：性伤害、暴力和错误信息。我们的研究结果表明，与单轮评估相比，多轮交互会导致缺陷率显着更高，凸显了VLLM中更深层次的漏洞。值得注意的是，根据我们的安全可用性指数（SUI）衡量，GPT-4 o表现出最平衡的性能，紧随其后的是Pixtral。此外，错误信息已成为一个需要加强上下文防御的关键领域。Llama-3.2表现出最高的MT缺陷率（16.55美元），而Qwen 2-BL表现出最高的MT拒绝率（19.1美元）。



## **19. ExpShield: Safeguarding Web Text from Unauthorized Crawling and Language Modeling Exploitation**

ExpShield：保护Web文本免受未经授权的抓取和语言建模利用 cs.CR

13 pages

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2412.21123v2) [paper-pdf](http://arxiv.org/pdf/2412.21123v2)

**Authors**: Ruixuan Liu, Toan Tran, Tianhao Wang, Hongsheng Hu, Shuo Wang, Li Xiong

**Abstract**: As large language models (LLMs) increasingly depend on web-scraped datasets, concerns arise over their potential to generate verbatim training content with copyrighted or private information. However, current protections against web crawling or sample-specific memorization are inherently limited, as they require compliance from crawlers (e.g., respecting robots.txt) or model trainers (e.g., applying differential privacy). To empower data owners with direct control, we propose ExpShiled, a proactive self-defense mechanism that mitigates sample-specific memorization via imperceptible text perturbations. This approach requires no external collaboration while maintaining original readability. To evaluate individual-level defense efficacy, we first propose the metric of instance exploitation: a zero value indicates perfect defense, achieved when a protected text's log-perplexity ranking aligns with its counterfactual untrained ranking. We then reveal and validate the memorization trigger hypothesis, demonstrating that a model's memorization of a specific text sample stems primarily from its outlier tokens. Leveraging this insight, we design targeted perturbations that (1) prioritize inherent trigger tokens and (2) introduce artificial trigger tokens as pitfalls to disrupt memorization on the protected sample. Experiments validate our defense across model scales, languages, vision-to-language tasks, and fine-tuning methods. Even with privacy backdoors, the Membership Inference Attack (MIA) AUC drops from 0.95 to 0.55, and instance exploitation approaches zero. This suggests that compared to the ideal no-misuse scenario, the risk of exposing a text instance remains nearly unchanged despite its inclusion in training data.

摘要: 随着大型语言模型（LLM）越来越依赖于网络抓取的数据集，人们开始担心它们是否有可能使用受版权或私人信息生成逐字训练内容。然而，当前针对网络爬行或特定样本记忆的保护本质上是有限的，因为它们需要爬行器的合规性（例如，尊重robots.文本）或模特教练（例如，应用差异隐私）。为了使数据所有者能够直接控制，我们提出了ExpShiled，这是一种主动的自我防御机制，可以通过不可察觉的文本扰动来减轻特定于样本的记忆。这种方法不需要外部协作，同时保持原始的可读性。为了评估个人级别的防御效率，我们首先提出了实例利用的度量：零值表示完美的防御，当受保护文本的日志困惑度排名与其反事实的未经训练的排名一致时。然后，我们揭示并验证了记忆触发假设，证明了模型对特定文本样本的记忆主要源于其离群值标记。利用这一洞察力，我们设计了有针对性的扰动，（1）优先考虑固有的触发令牌和（2）引入人工触发令牌作为陷阱，以破坏受保护样本的记忆。实验验证了我们的防御跨模型规模，语言，视觉到语言的任务，和微调方法。即使有隐私后门，成员推理攻击（MIA）AUC也从0.95下降到0.55，实例利用接近于零。这表明，与理想的无误用场景相比，尽管文本实例包含在训练数据中，但暴露文本实例的风险几乎保持不变。



## **20. Towards Universal and Black-Box Query-Response Only Attack on LLMs with QROA**

采用QROA对LLM进行通用和黑匣子仅查询响应攻击 cs.CL

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2406.02044v3) [paper-pdf](http://arxiv.org/pdf/2406.02044v3)

**Authors**: Hussein Jawad, Yassine Chenik, Nicolas J. -B. Brunel

**Abstract**: The rapid adoption of Large Language Models (LLMs) has exposed critical security and ethical vulnerabilities, particularly their susceptibility to adversarial manipulations. This paper introduces QROA, a novel black-box jailbreak method designed to identify adversarial suffixes that can bypass LLM alignment safeguards when appended to a malicious instruction. Unlike existing suffix-based jailbreak approaches, QROA does not require access to the model's logit or any other internal information. It also eliminates reliance on human-crafted templates, operating solely through the standard query-response interface of LLMs. By framing the attack as an optimization bandit problem, QROA employs a surrogate model and token level optimization to efficiently explore suffix variations. Furthermore, we propose QROA-UNV, an extension that identifies universal adversarial suffixes for individual models, enabling one-query jailbreaks across a wide range of instructions. Testing on multiple models demonstrates Attack Success Rate (ASR) greater than 80\%. These findings highlight critical vulnerabilities, emphasize the need for advanced defenses, and contribute to the development of more robust safety evaluations for secure AI deployment. The code is made public on the following link: https://github.com/qroa/QROA

摘要: 大型语言模型（LLM）的迅速采用暴露了关键的安全和道德漏洞，特别是它们容易受到对抗性操纵的影响。本文介绍了QROA，这是一种新型黑匣子越狱方法，旨在识别对抗性后缀，这些后缀在附加到恶意指令时可以绕过LLM对齐保障措施。与现有的基于后缀的越狱方法不同，QROA不需要访问模型的logit或任何其他内部信息。它还消除了对人工模板的依赖，仅通过LLM的标准查询-响应界面操作。通过将攻击定义为优化强盗问题，QROA采用代理模型和令牌级优化来有效地探索后缀变体。此外，我们还提出了QROA-UNV，这是一种扩展，可以为各个模型识别通用的对抗性后缀，从而实现跨广泛指令的单查询越狱。对多个模型的测试表明攻击成功率（ASB）大于80%。这些发现凸显了关键漏洞，强调了对先进防御的需要，并有助于开发更强大的安全评估以实现安全的人工智能部署。该代码在以下链接上公开：https://github.com/qroa/QROA



## **21. HAIR: Hardness-Aware Inverse Reinforcement Learning with Introspective Reasoning for LLM Alignment**

HAIR：具有内省推理的硬感知反向强化学习，用于LLM对齐 cs.CL

The three authors contributed equally to this work

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2503.18991v2) [paper-pdf](http://arxiv.org/pdf/2503.18991v2)

**Authors**: Ruoxi Cheng, Haoxuan Ma, Weixin Wang

**Abstract**: The alignment of large language models (LLMs) with human values remains critical yet hindered by four key challenges: (1) scarcity of balanced safety datasets, (2) alignment tax, (3) vulnerability to jailbreak attacks due to shallow alignment, and (4) inability to dynamically adapt rewards according to task difficulty. To address these limitations, we introduce HAIR (Hardness-Aware Inverse Reinforcement Learning with Introspective Reasoning), a novel alignment approach inspired by shadow models in membership inference attacks. Our approach consists of two main components: (1) construction of a balanced safety Chain-of-Draft (CoD) dataset for seven harmful categories using structured prompts that leverage the introspective reasoning capabilities of LLMs; and (2) training of category-specific reward models with Group Relative Policy Optimization (GRPO), dynamically tuning optimization to task difficulty at both the data and model levels. Comprehensive experiments across four harmlessness and four usefulness benchmarks demonstrate that HAIR achieves state-of-the-art performance, outperforming all baseline methods in safety while maintaining high levels of usefulness.

摘要: 大型语言模型（LLM）与人类价值观的一致仍然至关重要，但受到四个关键挑战的阻碍：（1）平衡安全数据集的稀缺，（2）一致税，（3）由于浅一致而容易受到越狱攻击，以及（4）无法根据任务难度动态调整奖励。为了解决这些局限性，我们引入了HAIR（具有内省推理的硬度感知反向强化学习），这是一种新颖的对齐方法，其灵感来自隶属推理攻击中的影子模型。我们的方法由两个主要部分组成：（1）使用利用LLM内省推理能力的结构化提示，为七个有害类别构建平衡的安全草案链（CoD）数据集;（2）使用组相对政策优化（GRPO）训练特定类别的奖励模型，动态调整优化以满足数据和模型级别的任务难度。四种无害性和四种有用性基准的综合实验表明，HAIR实现了最先进的性能，在安全性方面优于所有基线方法，同时保持了高水平的有用性。



## **22. BadLingual: A Novel Lingual-Backdoor Attack against Large Language Models**

BadLingual：针对大型语言模型的新型语言后门攻击 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03501v1) [paper-pdf](http://arxiv.org/pdf/2505.03501v1)

**Authors**: Zihan Wang, Hongwei Li, Rui Zhang, Wenbo Jiang, Kangjie Chen, Tianwei Zhang, Qingchuan Zhao, Guowen Xu

**Abstract**: In this paper, we present a new form of backdoor attack against Large Language Models (LLMs): lingual-backdoor attacks. The key novelty of lingual-backdoor attacks is that the language itself serves as the trigger to hijack the infected LLMs to generate inflammatory speech. They enable the precise targeting of a specific language-speaking group, exacerbating racial discrimination by malicious entities. We first implement a baseline lingual-backdoor attack, which is carried out by poisoning a set of training data for specific downstream tasks through translation into the trigger language. However, this baseline attack suffers from poor task generalization and is impractical in real-world settings. To address this challenge, we design BadLingual, a novel task-agnostic lingual-backdoor, capable of triggering any downstream tasks within the chat LLMs, regardless of the specific questions of these tasks. We design a new approach using PPL-constrained Greedy Coordinate Gradient-based Search (PGCG) based adversarial training to expand the decision boundary of lingual-backdoor, thereby enhancing the generalization ability of lingual-backdoor across various tasks. We perform extensive experiments to validate the effectiveness of our proposed attacks. Specifically, the baseline attack achieves an ASR of over 90% on the specified tasks. However, its ASR reaches only 37.61% across six tasks in the task-agnostic scenario. In contrast, BadLingual brings up to 37.35% improvement over the baseline. Our study sheds light on a new perspective of vulnerabilities in LLMs with multilingual capabilities and is expected to promote future research on the potential defenses to enhance the LLMs' robustness

摘要: 在本文中，我们提出了一种针对大型语言模型（LLM）的新形式后门攻击：语言后门攻击。语言后门攻击的关键新颖之处在于，语言本身充当了劫持受感染LLM以产生煽动性言语的触发器。它们能够准确瞄准特定语言群体，加剧恶意实体的种族歧视。我们首先实施基线语言后门攻击，通过翻译成触发语言来毒害特定下游任务的一组训练数据来执行该攻击。然而，这种基线攻击的任务概括性较差，并且在现实世界环境中不切实际。为了应对这一挑战，我们设计了BadLingual，这是一种新型的任务不可知语言后门，能够触发聊天LLM内的任何下游任务，无论这些任务的具体问题如何。我们设计了一种使用PPL约束的基于贪婪协调搜索（PGCG）的对抗训练的新方法，以扩大语言后门的决策边界，从而增强语言后门在各种任务中的概括能力。我们进行了广泛的实验来验证我们提出的攻击的有效性。具体来说，基线攻击在指定任务上实现了超过90%的ASB。然而，在任务不可知的场景中，其六项任务的ASB仅达到37.61%。相比之下，BadLingual较基线提高了37.35%。我们的研究揭示了具有多语言功能的LLM漏洞的新视角，并预计将促进未来对潜在防御措施的研究，以增强LLM的稳健性



## **23. Automatic Calibration for Membership Inference Attack on Large Language Models**

大型语言模型隶属度推理攻击的自动校准 cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03392v1) [paper-pdf](http://arxiv.org/pdf/2505.03392v1)

**Authors**: Saleh Zare Zade, Yao Qiang, Xiangyu Zhou, Hui Zhu, Mohammad Amin Roshani, Prashant Khanduri, Dongxiao Zhu

**Abstract**: Membership Inference Attacks (MIAs) have recently been employed to determine whether a specific text was part of the pre-training data of Large Language Models (LLMs). However, existing methods often misinfer non-members as members, leading to a high false positive rate, or depend on additional reference models for probability calibration, which limits their practicality. To overcome these challenges, we introduce a novel framework called Automatic Calibration Membership Inference Attack (ACMIA), which utilizes a tunable temperature to calibrate output probabilities effectively. This approach is inspired by our theoretical insights into maximum likelihood estimation during the pre-training of LLMs. We introduce ACMIA in three configurations designed to accommodate different levels of model access and increase the probability gap between members and non-members, improving the reliability and robustness of membership inference. Extensive experiments on various open-source LLMs demonstrate that our proposed attack is highly effective, robust, and generalizable, surpassing state-of-the-art baselines across three widely used benchmarks. Our code is available at: \href{https://github.com/Salehzz/ACMIA}{\textcolor{blue}{Github}}.

摘要: 成员资格推理攻击（MIA）最近被用来确定特定文本是否是大型语言模型（LLM）预训练数据的一部分。然而，现有方法经常将非成员误认为是成员，导致高假阳性率，或者依赖于额外的参考模型进行概率校准，这限制了其实用性。为了克服这些挑战，我们引入了一种名为自动校准成员推断攻击（ACMIA）的新型框架，该框架利用可调温度来有效地校准输出概率。这种方法的灵感来自我们对LLM预训练期间最大似然估计的理论见解。我们以三种配置引入ACMIA，旨在适应不同级别的模型访问并增加成员和非成员之间的概率差距，提高隶属推理的可靠性和稳健性。对各种开源LLM的广泛实验表明，我们提出的攻击非常有效、稳健且可推广，超越了三个广泛使用的基准测试中的最新基线。我们的代码可在以下网址获取：\href{https：//github.com/Salehzz/ACMIA}{\textColor{blue}{Github}}。



## **24. Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models**

使用机械可解释性来应对大型语言模型的对抗攻击 cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2503.06269v2) [paper-pdf](http://arxiv.org/pdf/2503.06269v2)

**Authors**: Thomas Winninger, Boussad Addad, Katarzyna Kapusta

**Abstract**: Traditional white-box methods for creating adversarial perturbations against LLMs typically rely only on gradient computation from the targeted model, ignoring the internal mechanisms responsible for attack success or failure. Conversely, interpretability studies that analyze these internal mechanisms lack practical applications beyond runtime interventions. We bridge this gap by introducing a novel white-box approach that leverages mechanistic interpretability techniques to craft practical adversarial inputs. Specifically, we first identify acceptance subspaces - sets of feature vectors that do not trigger the model's refusal mechanisms - then use gradient-based optimization to reroute embeddings from refusal subspaces to acceptance subspaces, effectively achieving jailbreaks. This targeted approach significantly reduces computation cost, achieving attack success rates of 80-95\% on state-of-the-art models including Gemma2, Llama3.2, and Qwen2.5 within minutes or even seconds, compared to existing techniques that often fail or require hours of computation. We believe this approach opens a new direction for both attack research and defense development. Furthermore, it showcases a practical application of mechanistic interpretability where other methods are less efficient, which highlights its utility. The code and generated datasets are available at https://github.com/Sckathach/subspace-rerouting.

摘要: 用于针对LLM创建对抗性扰动的传统白盒方法通常仅依赖于目标模型的梯度计算，而忽略了负责攻击成功或失败的内部机制。相反，分析这些内部机制的可解释性研究缺乏运行时干预之外的实际应用。我们通过引入一种新颖的白盒方法来弥合这一差距，该方法利用机械解释性技术来制作实用的对抗性输入。具体来说，我们首先识别接受子空间--不会触发模型拒绝机制的特征载体集--然后使用基于梯度的优化将嵌入从拒绝子空间重新路由到接受子空间，有效地实现越狱。与经常失败或需要数小时计算的现有技术相比，这种有针对性的方法显着降低了计算成本，在几分钟甚至几秒钟内就实现了对Gemma 2、Llama3.2和Qwen 2.5等最先进模型80- 95%的攻击成功率。我们相信这种方法为攻击研究和防御开发开辟了新的方向。此外，它展示了机械解释性的实际应用，而其他方法效率较低，这凸显了它的实用性。代码和生成的数据集可在https://github.com/Sckathach/subspace-rerouting上获取。



## **25. A Trustworthy Multi-LLM Network: Challenges,Solutions, and A Use Case**

值得信赖的多元LLM网络：挑战、解决方案和用例 cs.NI

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03196v1) [paper-pdf](http://arxiv.org/pdf/2505.03196v1)

**Authors**: Haoxiang Luo, Gang Sun, Yinqiu Liu, Dusit Niyato, Hongfang Yu, Mohammed Atiquzzaman, Schahram Dustdar

**Abstract**: Large Language Models (LLMs) demonstrate strong potential across a variety of tasks in communications and networking due to their advanced reasoning capabilities. However, because different LLMs have different model structures and are trained using distinct corpora and methods, they may offer varying optimization strategies for the same network issues. Moreover, the limitations of an individual LLM's training data, aggravated by the potential maliciousness of its hosting device, can result in responses with low confidence or even bias. To address these challenges, we propose a blockchain-enabled collaborative framework that connects multiple LLMs into a Trustworthy Multi-LLM Network (MultiLLMN). This architecture enables the cooperative evaluation and selection of the most reliable and high-quality responses to complex network optimization problems. Specifically, we begin by reviewing related work and highlighting the limitations of existing LLMs in collaboration and trust, emphasizing the need for trustworthiness in LLM-based systems. We then introduce the workflow and design of the proposed Trustworthy MultiLLMN framework. Given the severity of False Base Station (FBS) attacks in B5G and 6G communication systems and the difficulty of addressing such threats through traditional modeling techniques, we present FBS defense as a case study to empirically validate the effectiveness of our approach. Finally, we outline promising future research directions in this emerging area.

摘要: 大型语言模型（LLM）因其先进的推理能力而在通信和网络领域的各种任务中展现出强大的潜力。然而，由于不同的LLM具有不同的模型结构，并且使用不同的数据库和方法进行训练，因此它们可能会为相同的网络问题提供不同的优化策略。此外，个体LLM训练数据的局限性，再加上其托管设备的潜在恶意，可能会导致响应信心较低甚至有偏见。为了应对这些挑战，我们提出了一个支持区块链的协作框架，将多个LLM连接到值得信赖的多LLM网络（MultiLLNN）中。该架构能够对复杂的网络优化问题进行协作评估和选择最可靠和高质量的响应。具体来说，我们首先回顾相关工作，并强调现有的LLM在协作和信任方面的局限性，强调基于LLM的系统需要可信度。然后，我们介绍了工作流程和设计的建议值得信赖的MultiLLMN框架。鉴于B5G和6G通信系统中虚假基站（FBS）攻击的严重性以及通过传统建模技术解决此类威胁的困难，我们将FBS防御作为案例研究，以实证验证我们方法的有效性。最后，我们概述了这一新兴领域有前途的未来研究方向。



## **26. Towards Effective Identification of Attack Techniques in Cyber Threat Intelligence Reports using Large Language Models**

使用大型语言模型有效识别网络威胁情报报告中的攻击技术 cs.CR

5 pages, 2 figures 4 tables, accepted for publication at the Web  Conference 2025 (WWW'25)

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03147v1) [paper-pdf](http://arxiv.org/pdf/2505.03147v1)

**Authors**: Hoang Cuong Nguyen, Shahroz Tariq, Mohan Baruwal Chhetri, Bao Quoc Vo

**Abstract**: This work evaluates the performance of Cyber Threat Intelligence (CTI) extraction methods in identifying attack techniques from threat reports available on the web using the MITRE ATT&CK framework. We analyse four configurations utilising state-of-the-art tools, including the Threat Report ATT&CK Mapper (TRAM) and open-source Large Language Models (LLMs) such as Llama2. Our findings reveal significant challenges, including class imbalance, overfitting, and domain-specific complexity, which impede accurate technique extraction. To mitigate these issues, we propose a novel two-step pipeline: first, an LLM summarises the reports, and second, a retrained SciBERT model processes a rebalanced dataset augmented with LLM-generated data. This approach achieves an improvement in F1-scores compared to baseline models, with several attack techniques surpassing an F1-score of 0.90. Our contributions enhance the efficiency of web-based CTI systems and support collaborative cybersecurity operations in an interconnected digital landscape, paving the way for future research on integrating human-AI collaboration platforms.

摘要: 这项工作评估了网络威胁情报（RTI）提取方法在使用MITRE ATT & CK框架从网络上可用的威胁报告中识别攻击技术方面的性能。我们利用最先进的工具分析了四种配置，包括Threat Report ATT & CK Mapper（TRAM）和开源大型语言模型（LLM），例如Llama 2。我们的研究结果揭示了重大挑战，包括阶级不平衡、过度匹配和特定领域的复杂性，这些挑战阻碍了准确的技术提取。为了缓解这些问题，我们提出了一种新颖的两步管道：首先，LLM总结报告，其次，重新训练的SciBERT模型处理用LLM生成的数据增强的重新平衡数据集。与基线模型相比，这种方法提高了F1评分，其中几种攻击技术超过了F1评分0.90。我们的贡献提高了基于网络的RTI系统的效率，并支持互联数字环境中的协作网络安全操作，为未来整合人类与人工智能协作平台的研究铺平了道路。



## **27. PEEK: Phishing Evolution Framework for Phishing Generation and Evolving Pattern Analysis using Large Language Models**

TEK：使用大型语言模型进行网络钓鱼生成和演变模式分析的网络钓鱼进化框架 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2411.11389v2) [paper-pdf](http://arxiv.org/pdf/2411.11389v2)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), in particular, deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains detection effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems and people vulnerable to an ever-growing array of attacks. We propose the first Phishing Evolution FramEworK (PEEK) for augmenting phishing email datasets with respect to quality and diversity, and analyzing changing phishing patterns for detection to adapt to updated phishing attacks. Specifically, we integrate large language models (LLMs) into the process of adversarial training to enhance the performance of the generated dataset and leverage persuasion principles in a recurrent framework to facilitate the understanding of changing phishing strategies. PEEK raises the proportion of usable phishing samples from 21.4% to 84.8%, surpassing existing works that rely on prompting and fine-tuning LLMs. The phishing datasets provided by PEEK, with evolving phishing patterns, outperform the other two available LLM-generated phishing email datasets in improving detection robustness. PEEK phishing boosts detectors' accuracy to over 88% and reduces adversarial sensitivity by up to 70%, still maintaining 70% detection accuracy against adversarial attacks.

摘要: 网络钓鱼仍然是一种普遍存在的网络威胁，因为攻击者制作了欺骗性电子邮件来引诱受害者泄露敏感信息。虽然人工智能（AI），特别是深度学习，已成为防御网络钓鱼攻击的关键组成部分，但这些方法面临着严重的局限性。主要由于隐私问题，公开可用的、多样化的和更新的数据的稀缺限制了检测有效性。随着网络钓鱼策略的迅速发展，在有限、过时的数据上训练的模型很难检测到新的、复杂的欺骗策略，从而使系统和人们容易受到越来越多的攻击。我们提出了第一个网络钓鱼Evolution FramEworK（TEK），用于增强网络钓鱼电子邮件数据集的质量和多样性，并分析不断变化的网络钓鱼模式进行检测，以适应更新的网络钓鱼攻击。具体来说，我们将大型语言模型（LLM）集成到对抗训练过程中，以增强生成的数据集的性能，并在循环框架中利用说服原则，以促进对不断变化的网络钓鱼策略的理解。TEK将可用网络钓鱼样本的比例从21.4%提高到84.8%，超过了依赖提示和微调LLM的现有作品。TEK提供的网络钓鱼数据集具有不断变化的网络钓鱼模式，在提高检测稳健性方面优于其他两个可用的LLM生成的网络钓鱼电子邮件数据集。TEK网络钓鱼将检测器的准确性提高至88%以上，并将对抗敏感性降低高达70%，针对对抗攻击仍保持70%的检测准确性。



## **28. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

大型语言模型作为软件分析中稳健的数据生成器：我们已经做到了吗？ cs.SE

Accepted to the AI Model/Data Track of the Evaluation and Assessment  in Software Engineering (EASE) 2025 Conference

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2411.10565v3) [paper-pdf](http://arxiv.org/pdf/2411.10565v3)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.

摘要: 大型语言模型（LLM）生成的数据越来越多地用于软件分析，但目前尚不清楚该数据与人类编写的数据相比如何，特别是当模型暴露于对抗场景时。对抗性攻击可能会损害软件系统的可靠性和安全性，因此，与作为模型性能基准的人类编写数据相比，了解LLM生成的数据在这些条件下的表现如何，可以为LLM生成的数据是否提供类似的稳健性和有效性提供有价值的见解。为了解决这一差距，我们系统地评估和比较人类编写的数据和LLM生成的数据的质量，以便在对抗性攻击的背景下微调稳健的预训练模型（Ptms）。我们评估了六种广泛使用的PtM的稳健性，这些PtM在对抗性攻击之前和之后根据人类编写和LLM生成的数据进行了微调。该评估在三个流行的软件分析任务中使用了九种最先进的（SOTA）对抗性攻击技术：克隆检测，代码摘要和代码审查讨论中的情感分析。此外，我们使用11个相似性度量来分析生成的对抗性示例的质量。我们的研究结果表明，虽然对LLM生成的数据进行微调的PTM与对人类编写的数据进行微调的PTM具有竞争力，但它们在软件分析任务中对对抗性攻击的鲁棒性较低。我们的研究强调了进一步探索提高LLM生成的训练数据质量的必要性，以开发高性能且能够抵御软件分析中的对抗性攻击的模型。



## **29. Large Language Models as Carriers of Hidden Messages**

大型语言模型作为隐藏消息的载体 cs.CL

Accepted on SECRYPT 2025 Conference. Code is available at  https://github.com/j-hoscilowic/zurek-stegano

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2406.02481v5) [paper-pdf](http://arxiv.org/pdf/2406.02481v5)

**Authors**: Jakub Hoscilowicz, Pawel Popiolek, Jan Rudkowski, Jedrzej Bieniasz, Artur Janicki

**Abstract**: Simple fine-tuning can embed hidden text into large language models (LLMs), which is revealed only when triggered by a specific query. Applications include LLM fingerprinting, where a unique identifier is embedded to verify licensing compliance, and steganography, where the LLM carries hidden messages disclosed through a trigger query.   Our work demonstrates that embedding hidden text via fine-tuning, although seemingly secure due to the vast number of potential triggers, is vulnerable to extraction through analysis of the LLM's output decoding process. We introduce an extraction attack called Unconditional Token Forcing (UTF), which iteratively feeds tokens from the LLM's vocabulary to reveal sequences with high token probabilities, indicating hidden text candidates. We also present Unconditional Token Forcing Confusion (UTFC), a defense paradigm that makes hidden text resistant to all known extraction attacks without degrading the general performance of LLMs compared to standard fine-tuning. UTFC has both benign (improving LLM fingerprinting) and malign applications (using LLMs to create covert communication channels).

摘要: 简单的微调可以将隐藏文本嵌入到大型语言模型（LLM）中，只有在特定查询触发时才会显示该文本。应用包括LLM指纹识别（其中嵌入唯一标识符以验证许可合规性）和隐写术（其中LLM携带通过触发查询披露的隐藏消息）。   我们的工作表明，通过微调嵌入隐藏文本，尽管由于潜在触发器数量庞大，看起来很安全，但通过分析LLM的输出解码过程，很容易被提取。我们引入了一种名为无条件令牌强迫（UTF）的提取攻击，它迭代地从LLM词汇表中输入令牌，以揭示具有高令牌概率的序列，从而指示隐藏的文本候选项。我们还介绍了无条件令牌强制混淆（UTFC），这是一种防御范式，可以使隐藏文本抵御所有已知的提取攻击，而与标准微调相比，不会降低LLM的总体性能。UTFC既有良性应用程序（改进LLM指纹识别），也有恶意应用程序（使用LLM创建秘密通信渠道）。



## **30. Mapping the Italian Telegram Ecosystem: Communities, Toxicity, and Hate Speech**

绘制意大利电报生态系统：社区，毒性和仇恨言论 cs.SI

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2504.19594v2) [paper-pdf](http://arxiv.org/pdf/2504.19594v2)

**Authors**: Lorenzo Alvisi, Serena Tardelli, Maurizio Tesconi

**Abstract**: Telegram has become a major space for political discourse and alternative media. However, its lack of moderation allows misinformation, extremism, and toxicity to spread. While prior research focused on these particular phenomena or topics, these have mostly been examined separately, and a broader understanding of the Telegram ecosystem is still missing. In this work, we fill this gap by conducting a large-scale analysis of the Italian Telegram sphere, leveraging a dataset of 186 million messages from 13,151 chats collected in 2023. Using network analysis, Large Language Models, and toxicity detection tools, we examine how different thematic communities form, align ideologically, and engage in harmful discourse within the Italian cultural context. Results show strong thematic and ideological homophily. We also identify mixed ideological communities where far-left and far-right rhetoric coexist on particular geopolitical issues. Beyond political analysis, we find that toxicity, rather than being isolated in a few extreme chats, appears widely normalized within highly toxic communities. Moreover, we find that Italian discourse primarily targets Black people, Jews, and gay individuals independently of the topic. Finally, we uncover common trend of intra-national hostility, where Italians often attack other Italians, reflecting regional and intra-regional cultural conflicts that can be traced back to old historical divisions. This study provides the first large-scale mapping of the Italian Telegram ecosystem, offering insights into ideological interactions, toxicity, and identity-targets of hate and contributing to research on online toxicity across different cultural and linguistic contexts on Telegram.

摘要: Telegram已成为政治话语和另类媒体的主要空间。然而，它缺乏节制，导致错误信息、极端主义和毒性蔓延。虽然之前的研究集中在这些特定的现象或主题上，但这些主要是单独研究的，并且仍然缺乏对Telegram生态系统的更广泛的了解。在这项工作中，我们通过对意大利Telegram领域进行大规模分析来填补这一空白，利用2023年收集的13，151条聊天记录中的1.86亿条消息的数据集。使用网络分析、大型语言模型和毒性检测工具，我们研究不同的主题社区如何在意大利文化背景下形成、意识形态上的一致以及参与有害话语。结果显示出较强的主题和意识形态一致性。我们还发现了混合的意识形态社区，其中极左和极右言论在特定地缘政治问题上共存。除了政治分析之外，我们发现毒性并没有在一些极端的聊天中被孤立，而是在高毒性社区中被广泛正常化。此外，我们发现意大利语的话语主要针对黑人、犹太人和同性恋者，与主题无关。最后，我们发现了国内敌意的共同趋势，意大利人经常攻击其他意大利人，反映了可以追溯到旧历史分歧的地区和地区内文化冲突。这项研究首次对意大利Telegram生态系统进行了大规模映射，提供了对意识形态相互作用、毒性和仇恨身份目标的见解，并为Telegram上不同文化和语言背景下的在线毒性研究做出了贡献。



## **31. A Survey on Privacy Risks and Protection in Large Language Models**

大型语言模型中的隐私风险与保护调查 cs.CR

**SubmitDate**: 2025-05-04    [abs](http://arxiv.org/abs/2505.01976v1) [paper-pdf](http://arxiv.org/pdf/2505.01976v1)

**Authors**: Kang Chen, Xiuze Zhou, Yuanguo Lin, Shibo Feng, Li Shen, Pengcheng Wu

**Abstract**: Although Large Language Models (LLMs) have become increasingly integral to diverse applications, their capabilities raise significant privacy concerns. This survey offers a comprehensive overview of privacy risks associated with LLMs and examines current solutions to mitigate these challenges. First, we analyze privacy leakage and attacks in LLMs, focusing on how these models unintentionally expose sensitive information through techniques such as model inversion, training data extraction, and membership inference. We investigate the mechanisms of privacy leakage, including the unauthorized extraction of training data and the potential exploitation of these vulnerabilities by malicious actors. Next, we review existing privacy protection against such risks, such as inference detection, federated learning, backdoor mitigation, and confidential computing, and assess their effectiveness in preventing privacy leakage. Furthermore, we highlight key practical challenges and propose future research directions to develop secure and privacy-preserving LLMs, emphasizing privacy risk assessment, secure knowledge transfer between models, and interdisciplinary frameworks for privacy governance. Ultimately, this survey aims to establish a roadmap for addressing escalating privacy challenges in the LLMs domain.

摘要: 虽然大型语言模型（LLM）已经成为各种应用程序的一部分，但它们的功能引起了严重的隐私问题。该调查全面概述了与LLM相关的隐私风险，并研究了缓解这些挑战的当前解决方案。首先，我们分析了LLM中的隐私泄露和攻击，重点关注这些模型如何通过模型反演、训练数据提取和成员推断等技术无意中暴露敏感信息。我们调查了隐私泄露的机制，包括未经授权提取训练数据以及恶意行为者可能利用这些漏洞。接下来，我们回顾针对此类风险（例如推断检测、联邦学习、后门缓解和机密计算）的现有隐私保护，并评估它们在防止隐私泄露方面的有效性。此外，我们还强调了关键的实践挑战，并提出了未来的研究方向，以开发安全且保护隐私的LLM，强调隐私风险评估、模型之间的安全知识转移以及隐私治理的跨学科框架。最终，这项调查旨在建立一个路线图，以解决LLM领域不断升级的隐私挑战。



## **32. Cannot See the Forest for the Trees: Invoking Heuristics and Biases to Elicit Irrational Choices of LLMs**

只见树木不见森林：利用启发式和偏见来激发对LLM的非理性选择 cs.CL

**SubmitDate**: 2025-05-03    [abs](http://arxiv.org/abs/2505.02862v1) [paper-pdf](http://arxiv.org/pdf/2505.02862v1)

**Authors**: Haoming Yang, Ke Ma, Xiaojun Jia, Yingfei Sun, Qianqian Xu, Qingming Huang

**Abstract**: Despite the remarkable performance of Large Language Models (LLMs), they remain vulnerable to jailbreak attacks, which can compromise their safety mechanisms. Existing studies often rely on brute-force optimization or manual design, failing to uncover potential risks in real-world scenarios. To address this, we propose a novel jailbreak attack framework, ICRT, inspired by heuristics and biases in human cognition. Leveraging the simplicity effect, we employ cognitive decomposition to reduce the complexity of malicious prompts. Simultaneously, relevance bias is utilized to reorganize prompts, enhancing semantic alignment and inducing harmful outputs effectively. Furthermore, we introduce a ranking-based harmfulness evaluation metric that surpasses the traditional binary success-or-failure paradigm by employing ranking aggregation methods such as Elo, HodgeRank, and Rank Centrality to comprehensively quantify the harmfulness of generated content. Experimental results show that our approach consistently bypasses mainstream LLMs' safety mechanisms and generates high-risk content, providing insights into jailbreak attack risks and contributing to stronger defense strategies.

摘要: 尽管大型语言模型（LLM）性能出色，但它们仍然容易受到越狱攻击，这可能会损害其安全机制。现有的研究通常依赖于暴力优化或手动设计，未能发现现实世界场景中的潜在风险。为了解决这个问题，我们提出了一种新颖的越狱攻击框架ICRT，其灵感来自人类认知中的启发和偏见。利用简单性效应，我们采用认知分解来降低恶意提示的复杂性。同时，利用相关性偏差来重组提示，增强语义对齐并有效地诱导有害输出。此外，我们引入了一种基于排名的危害性评估指标，通过采用Elo、HodgeRank和Rank Centrality等排名聚合方法来全面量化生成内容的危害性，超越了传统的二元成败范式。实验结果表明，我们的方法始终绕过主流LLM的安全机制并生成高风险内容，提供了对越狱攻击风险的见解，并有助于制定更强有力的防御策略。



## **33. Parameterized Argumentation-based Reasoning Tasks for Benchmarking Generative Language Models**

用于生成语言模型基准测试的参数化基于论证的推理任务 cs.AI

This manuscript has been accepted for presentation as a short paper  at the 20th International Conference of AI & Law in Chicago, June 16 to 20 of  2025

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01539v1) [paper-pdf](http://arxiv.org/pdf/2505.01539v1)

**Authors**: Cor Steging, Silja Renooij, Bart Verheij

**Abstract**: Generative large language models as tools in the legal domain have the potential to improve the justice system. However, the reasoning behavior of current generative models is brittle and poorly understood, hence cannot be responsibly applied in the domains of law and evidence. In this paper, we introduce an approach for creating benchmarks that can be used to evaluate the reasoning capabilities of generative language models. These benchmarks are dynamically varied, scalable in their complexity, and have formally unambiguous interpretations. In this study, we illustrate the approach on the basis of witness testimony, focusing on the underlying argument attack structure. We dynamically generate both linear and non-linear argument attack graphs of varying complexity and translate these into reasoning puzzles about witness testimony expressed in natural language. We show that state-of-the-art large language models often fail in these reasoning puzzles, already at low complexity. Obvious mistakes are made by the models, and their inconsistent performance indicates that their reasoning capabilities are brittle. Furthermore, at higher complexity, even state-of-the-art models specifically presented for reasoning capabilities make mistakes. We show the viability of using a parametrized benchmark with varying complexity to evaluate the reasoning capabilities of generative language models. As such, the findings contribute to a better understanding of the limitations of the reasoning capabilities of generative models, which is essential when designing responsible AI systems in the legal domain.

摘要: 生成性大型语言模型作为法律领域的工具有潜力改善司法系统。然而，当前生成模型的推理行为很脆弱，而且人们理解得很差，因此无法负责任地应用于法律和证据领域。在本文中，我们介绍了一种创建基准的方法，该基准可用于评估生成式语言模型的推理能力。这些基准是动态变化的，其复杂性可扩展的，并且具有形式上明确的解释。在这项研究中，我们根据证人证词说明了这种方法，重点关注潜在的论点攻击结构。我们动态生成复杂性不同的线性和非线性论点攻击图，并将其转化为有关用自然语言表达的证人证词的推理难题。我们表明，最先进的大型语言模型经常在这些推理难题中失败，而且复杂性已经很低。模型会犯明显的错误，而且它们的不一致的性能表明它们的推理能力很脆弱。此外，在更高的复杂性下，即使是专门为推理能力提供的最先进的模型也会出错。我们展示了使用具有不同复杂性的参数化基准来评估生成式语言模型的推理能力的可行性。因此，这些发现有助于更好地理解生成模型推理能力的局限性，这在法律领域设计负责任的人工智能系统时至关重要。



## **34. Rubber Mallet: A Study of High Frequency Localized Bit Flips and Their Impact on Security**

橡皮锤：高频局部位翻转及其对安全性影响的研究 cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01518v1) [paper-pdf](http://arxiv.org/pdf/2505.01518v1)

**Authors**: Andrew Adiletta, Zane Weissman, Fatemeh Khojasteh Dana, Berk Sunar, Shahin Tajik

**Abstract**: The increasing density of modern DRAM has heightened its vulnerability to Rowhammer attacks, which induce bit flips by repeatedly accessing specific memory rows. This paper presents an analysis of bit flip patterns generated by advanced Rowhammer techniques that bypass existing hardware defenses. First, we investigate the phenomenon of adjacent bit flips--where two or more physically neighboring bits are corrupted simultaneously--and demonstrate they occur with significantly higher frequency than previously documented. We also show that if multiple bits flip within a byte, they are more likely to be adjacent than randomly distributed: for example, if 4 bits flip within a byte, there is an 87% chance that they are all adjacent. We also demonstrate that bit flips within a row will naturally cluster together likely due to the underlying physics of the attack. We then investigate two fault injection attacks enabled by multiple adjacent or nearby bit flips. First, we show how these correlated flips enable efficient cryptographic signature correction attacks, successfully recovering ECDSA private keys from OpenSSL implementations where single-bit approaches would be unfeasible. Second, we introduce a targeted attack against large language models by exploiting Rowhammer-induced corruptions in tokenizer dictionaries of GGUF model files. This attack effectively rewrites safety instructions in system prompts by swapping safety-critical tokens with benign alternatives, circumventing model guardrails while maintaining normal functionality in other contexts. Our experimental results across multiple DRAM configurations reveal that current memory protection schemes are inadequate against these sophisticated attack vectors, which can achieve their objectives with precise, minimal modifications rather than random corruption.

摘要: 现代动态存储器密度的增加加剧了其对Rowhammer攻击的脆弱性，Rowhammer攻击通过重复访问特定内存行来引发位翻转。本文分析了绕过现有硬件防御的高级Rowhammer技术生成的位翻转模式。首先，我们研究相邻位翻转的现象--两个或更多物理相邻位同时被破坏--并证明它们的发生频率比之前记录的要高得多。我们还表明，如果多个比特在一个字节内翻转，那么它们更有可能相邻而不是随机分布：例如，如果一个字节内翻转4个比特，那么它们都相邻的可能性有87%。我们还证明，由于攻击的基本物理原理，行内的位翻转可能会自然聚集在一起。然后，我们研究由多个相邻或附近的位翻转引发的两种故障注入攻击。首先，我们展示了这些相关翻转如何实现高效的加密签名纠正攻击，成功从单位方法不可行的OpenSSL实现中恢复ECDSA私有密钥。其次，我们通过利用GGUF模型文件的标记器字典中Rowhammer引起的损坏，引入针对大型语言模型的有针对性的攻击。这种攻击通过将安全关键令牌与良性替代方案交换，有效地重写了系统提示中的安全指令，绕过模型护栏，同时在其他上下文中保持正常功能。我们跨多种RAM配置的实验结果表明，当前的内存保护方案不足以对抗这些复杂的攻击载体，这些攻击载体可以通过精确、最少的修改而不是随机损坏来实现其目标。



## **35. LLM Security: Vulnerabilities, Attacks, Defenses, and Countermeasures**

LLM安全：漏洞、攻击、防御和对策 cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01177v1) [paper-pdf](http://arxiv.org/pdf/2505.01177v1)

**Authors**: Francisco Aguilera-Martínez, Fernando Berzal

**Abstract**: As large language models (LLMs) continue to evolve, it is critical to assess the security threats and vulnerabilities that may arise both during their training phase and after models have been deployed. This survey seeks to define and categorize the various attacks targeting LLMs, distinguishing between those that occur during the training phase and those that affect already trained models. A thorough analysis of these attacks is presented, alongside an exploration of defense mechanisms designed to mitigate such threats. Defenses are classified into two primary categories: prevention-based and detection-based defenses. Furthermore, our survey summarizes possible attacks and their corresponding defense strategies. It also provides an evaluation of the effectiveness of the known defense mechanisms for the different security threats. Our survey aims to offer a structured framework for securing LLMs, while also identifying areas that require further research to improve and strengthen defenses against emerging security challenges.

摘要: 随着大型语言模型（LLM）的不断发展，评估在训练阶段和模型部署后可能出现的安全威胁和漏洞至关重要。本调查旨在定义和分类针对LLM的各种攻击，区分那些在训练阶段发生的攻击和那些影响已经训练好的模型的攻击。本文对这些攻击进行了全面的分析，并探讨了旨在减轻此类威胁的防御机制。防御分为两大类：基于预防的防御和基于检测的防御。此外，我们的调查还总结了可能的攻击及其相应的防御策略。它还评估了已知防御机制对不同安全威胁的有效性。我们的调查旨在提供一个结构化的框架来保护法学硕士，同时确定需要进一步研究的领域，以改善和加强对新兴安全挑战的防御。



## **36. A Rusty Link in the AI Supply Chain: Detecting Evil Configurations in Model Repositories**

AI供应链中的生锈链接：检测模型库中的恶意行为 cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01067v1) [paper-pdf](http://arxiv.org/pdf/2505.01067v1)

**Authors**: Ziqi Ding, Qian Fu, Junchen Ding, Gelei Deng, Yi Liu, Yuekang Li

**Abstract**: Recent advancements in large language models (LLMs) have spurred the development of diverse AI applications from code generation and video editing to text generation; however, AI supply chains such as Hugging Face, which host pretrained models and their associated configuration files contributed by the public, face significant security challenges; in particular, configuration files originally intended to set up models by specifying parameters and initial settings can be exploited to execute unauthorized code, yet research has largely overlooked their security compared to that of the models themselves; in this work, we present the first comprehensive study of malicious configurations on Hugging Face, identifying three attack scenarios (file, website, and repository operations) that expose inherent risks; to address these threats, we introduce CONFIGSCAN, an LLM-based tool that analyzes configuration files in the context of their associated runtime code and critical libraries, effectively detecting suspicious elements with low false positive rates and high accuracy; our extensive evaluation uncovers thousands of suspicious repositories and configuration files, underscoring the urgent need for enhanced security validation in AI model hosting platforms.

摘要: 大型语言模型（LLM）的最新进展刺激了从代码生成、视频编辑到文本生成等各种人工智能应用的发展;然而，拥抱脸等人工智能供应链（托管预训练模型及其相关配置文件）面临着重大的安全挑战;公众贡献;特别是，最初旨在通过指定参数和初始设置来建立模型的配置文件可能会被利用来执行未经授权的代码，然而，与模型本身相比，研究在很大程度上忽视了它们的安全性;在这项工作中，我们首次对Hugging Face上的恶意配置进行了全面研究，识别了三种攻击场景暴露固有风险的（文件、网站和存储库操作）;为了解决这些威胁，我们引入了CONFIGSCAN，这是一种基于LLM的工具，可以在相关的运行时代码和关键库的上下文中分析配置文件，以低误报率和高准确性有效检测可疑元素;我们的广泛评估发现了数千个可疑存储库和配置文件，凸显了人工智能模型托管平台中增强安全验证的迫切需要。



## **37. Good News for Script Kiddies? Evaluating Large Language Models for Automated Exploit Generation**

好消息给孩子们。评估大型语言模型以自动生成漏洞 cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01065v1) [paper-pdf](http://arxiv.org/pdf/2505.01065v1)

**Authors**: David Jin, Qian Fu, Yuekang Li

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in code-related tasks, raising concerns about their potential for automated exploit generation (AEG). This paper presents the first systematic study on LLMs' effectiveness in AEG, evaluating both their cooperativeness and technical proficiency. To mitigate dataset bias, we introduce a benchmark with refactored versions of five software security labs. Additionally, we design an LLM-based attacker to systematically prompt LLMs for exploit generation. Our experiments reveal that GPT-4 and GPT-4o exhibit high cooperativeness, comparable to uncensored models, while Llama3 is the most resistant. However, no model successfully generates exploits for refactored labs, though GPT-4o's minimal errors highlight the potential for LLM-driven AEG advancements.

摘要: 大型语言模型（LLM）在代码相关任务中表现出了非凡的能力，这引发了人们对其自动利用生成（AEG）潜力的担忧。本文首次对LLM在AEG中的有效性进行了系统研究，评估了他们的合作性和技术熟练程度。为了减轻数据集偏见，我们引入了一个具有五个软件安全实验室重构版本的基准测试。此外，我们设计了一个基于LLM的攻击者来系统性地提示LLM进行漏洞利用生成。我们的实验表明，GPT-4和GPT-4 o表现出高度的协作性，与未经审查的模型相当，而Llama 3的抵抗力最强。然而，没有一个模型成功地为重构实验室生成漏洞，尽管GPT-4 o的最小错误凸显了LLM驱动的AEG进步的潜力。



## **38. Transferable Adversarial Attacks on Black-Box Vision-Language Models**

黑匣子视觉语言模型的可转移对抗攻击 cs.CV

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01050v1) [paper-pdf](http://arxiv.org/pdf/2505.01050v1)

**Authors**: Kai Hu, Weichen Yu, Li Zhang, Alexander Robey, Andy Zou, Chengming Xu, Haoqi Hu, Matt Fredrikson

**Abstract**: Vision Large Language Models (VLLMs) are increasingly deployed to offer advanced capabilities on inputs comprising both text and images. While prior research has shown that adversarial attacks can transfer from open-source to proprietary black-box models in text-only and vision-only contexts, the extent and effectiveness of such vulnerabilities remain underexplored for VLLMs. We present a comprehensive analysis demonstrating that targeted adversarial examples are highly transferable to widely-used proprietary VLLMs such as GPT-4o, Claude, and Gemini. We show that attackers can craft perturbations to induce specific attacker-chosen interpretations of visual information, such as misinterpreting hazardous content as safe, overlooking sensitive or restricted material, or generating detailed incorrect responses aligned with the attacker's intent. Furthermore, we discover that universal perturbations -- modifications applicable to a wide set of images -- can consistently induce these misinterpretations across multiple proprietary VLLMs. Our experimental results on object recognition, visual question answering, and image captioning show that this vulnerability is common across current state-of-the-art models, and underscore an urgent need for robust mitigations to ensure the safe and secure deployment of VLLMs.

摘要: Vision大型语言模型（VLLM）越来越多地部署，以提供包含文本和图像的输入的高级功能。虽然之前的研究表明，在纯文本和纯视觉的环境中，对抗性攻击可以从开源模型转移到专有黑匣子模型，但对于VLLM来说，此类漏洞的范围和有效性仍然没有得到充分的研究。我们提出了一项全面的分析，证明有针对性的对抗性示例可以高度转移到广泛使用的专有VLLM，例如GPT-4 o、Claude和Gemini。我们表明，攻击者可以制造干扰来诱导攻击者选择的特定视觉信息解释，例如将危险内容误解为安全内容、忽略敏感或受限制材料，或者生成与攻击者意图一致的详细错误响应。此外，我们发现普遍扰动（适用于广泛图像集的修改）可能会在多个专有VLLM中持续引发这些误解。我们关于对象识别、视觉问答和图像字幕的实验结果表明，这种漏洞在当前最先进的模型中很常见，并强调迫切需要强大的缓解措施，以确保VLLM的安全部署。



## **39. Prompt Inversion Attack against Collaborative Inference of Large Language Models**

针对大型语言模型协作推理的提示倒置攻击 cs.CR

To appear at IEEE Symposium on Security and Privacy 2025

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2503.09022v3) [paper-pdf](http://arxiv.org/pdf/2503.09022v3)

**Authors**: Wenjie Qu, Yuguang Zhou, Yongji Wu, Tingsong Xiao, Binhang Yuan, Yiming Li, Jiaheng Zhang

**Abstract**: Large language models (LLMs) have been widely applied for their remarkable capability of content generation. However, the practical use of open-source LLMs is hindered by high resource requirements, making deployment expensive and limiting widespread development. The collaborative inference is a promising solution for this problem, in which users collaborate by each hosting a subset of layers and transmitting intermediate activation. Many companies are building collaborative inference platforms to reduce LLM serving costs, leveraging users' underutilized GPUs. Despite widespread interest in collaborative inference within academia and industry, the privacy risks associated with LLM collaborative inference have not been well studied. This is largely because of the challenge posed by inverting LLM activation due to its strong non-linearity.   In this paper, to validate the severity of privacy threats in LLM collaborative inference, we introduce the concept of prompt inversion attack (PIA), where a malicious participant intends to recover the input prompt through the activation transmitted by its previous participant. Extensive experiments show that our PIA method substantially outperforms existing baselines. For example, our method achieves an 88.4\% token accuracy on the Skytrax dataset with the Llama-65B model when inverting the maximum number of transformer layers, while the best baseline method only achieves 22.8\% accuracy. The results verify the effectiveness of our PIA attack and highlights its practical threat to LLM collaborative inference systems.

摘要: 大型语言模型（LLM）因其出色的内容生成能力而得到广泛应用。然而，开源LLM的实际使用受到高资源需求的阻碍，导致部署成本高昂并限制了广泛开发。协作推理是这个问题的一个有希望的解决方案，其中用户通过各自托管层的子集并传输中间激活来进行协作。许多公司正在构建协作推理平台，以利用用户未充分利用的图形处理器来降低LLM服务成本。尽管学术界和工业界对协作推理产生了广泛的兴趣，但与LLM协作推理相关的隐私风险尚未得到很好的研究。这主要是因为LLM激活由于其强非线性而带来的挑战。   本文中，为了验证LLM协同推理中隐私威胁的严重性，我们引入了提示倒置攻击（PIA）的概念，恶意参与者意图通过其前一参与者传输的激活来恢复输入提示。大量实验表明，我们的PIA方法的性能大大优于现有的基线。例如，当倒置最大数量的Transformer层时，我们的方法在使用Llama-65 B模型的Skytrax数据集上实现了88.4%的令牌准确性，而最佳基线方法仅实现了22.8%的准确性。结果验证了我们PIA攻击的有效性，并强调了其对LLM协同推理系统的实际威胁。



## **40. Attack and defense techniques in large language models: A survey and new perspectives**

大型语言模型中的攻击和防御技术：概览和新观点 cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.00976v1) [paper-pdf](http://arxiv.org/pdf/2505.00976v1)

**Authors**: Zhiyu Liao, Kang Chen, Yuanguo Lin, Kangkang Li, Yunxuan Liu, Hefeng Chen, Xingwang Huang, Yuanhui Yu

**Abstract**: Large Language Models (LLMs) have become central to numerous natural language processing tasks, but their vulnerabilities present significant security and ethical challenges. This systematic survey explores the evolving landscape of attack and defense techniques in LLMs. We classify attacks into adversarial prompt attack, optimized attacks, model theft, as well as attacks on application of LLMs, detailing their mechanisms and implications. Consequently, we analyze defense strategies, including prevention-based and detection-based defense methods. Although advances have been made, challenges remain to adapt to the dynamic threat landscape, balance usability with robustness, and address resource constraints in defense implementation. We highlight open problems, including the need for adaptive scalable defenses, explainable security techniques, and standardized evaluation frameworks. This survey provides actionable insights and directions for developing secure and resilient LLMs, emphasizing the importance of interdisciplinary collaboration and ethical considerations to mitigate risks in real-world applications.

摘要: 大型语言模型（LLM）已成为众多自然语言处理任务的核心，但它们的漏洞带来了重大的安全和道德挑战。这项系统性调查探讨了LLM攻击和防御技术不断变化的格局。我们将攻击分为对抗即时攻击、优化攻击、模型盗窃以及对LLM应用程序的攻击，详细介绍了它们的机制和含义。因此，我们分析防御策略，包括基于预防和基于检测的防御方法。虽然已经取得了进展，但挑战仍然存在，以适应动态的威胁环境，平衡可用性与鲁棒性，并解决防御实施中的资源限制。我们强调开放的问题，包括需要自适应可扩展的防御，可解释的安全技术，和标准化的评估框架。该调查为开发安全和弹性LLM提供了可操作的见解和方向，强调了跨学科合作和道德考虑的重要性，以减轻现实世界应用中的风险。



## **41. Protocol-agnostic and Data-free Backdoor Attacks on Pre-trained Models in RF Fingerprinting**

对RF指纹识别中预训练模型的协议不可知且无数据后门攻击 cs.CR

10 pages, 7 figures, accepted by IEEE INFOCOM 2025

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00881v1) [paper-pdf](http://arxiv.org/pdf/2505.00881v1)

**Authors**: Tianya Zhao, Ningning Wang, Junqing Zhang, Xuyu Wang

**Abstract**: While supervised deep neural networks (DNNs) have proven effective for device authentication via radio frequency (RF) fingerprinting, they are hindered by domain shift issues and the scarcity of labeled data. The success of large language models has led to increased interest in unsupervised pre-trained models (PTMs), which offer better generalization and do not require labeled datasets, potentially addressing the issues mentioned above. However, the inherent vulnerabilities of PTMs in RF fingerprinting remain insufficiently explored. In this paper, we thoroughly investigate data-free backdoor attacks on such PTMs in RF fingerprinting, focusing on a practical scenario where attackers lack access to downstream data, label information, and training processes. To realize the backdoor attack, we carefully design a set of triggers and predefined output representations (PORs) for the PTMs. By mapping triggers and PORs through backdoor training, we can implant backdoor behaviors into the PTMs, thereby introducing vulnerabilities across different downstream RF fingerprinting tasks without requiring prior knowledge. Extensive experiments demonstrate the wide applicability of our proposed attack to various input domains, protocols, and PTMs. Furthermore, we explore potential detection and defense methods, demonstrating the difficulty of fully safeguarding against our proposed backdoor attack.

摘要: 虽然监督式深度神经网络（DNN）已被证明对于通过射频（RF）指纹识别的设备认证有效，但它们受到域转移问题和标记数据稀缺的阻碍。大型语言模型的成功导致了人们对无监督预训练模型（Ptms）的兴趣越来越大，这些模型提供更好的概括性，并且不需要标记的数据集，从而可能解决上述问题。然而，RF指纹识别中的PTO固有漏洞仍然没有得到充分的探索。在本文中，我们彻底调查了对RF指纹识别中此类PTO的无数据后门攻击，重点关注攻击者无法访问下游数据、标签信息和训练过程的实际场景。为了实现后门攻击，我们精心设计了一组触发器和预定义的输出表示（POL）。通过通过后门训练映射触发器和POL，我们可以将后门行为植入到PTO中，从而在不需要先验知识的情况下在不同下游RF指纹识别任务中引入漏洞。大量的实验证明了我们提出的攻击对各种输入域、协议和STM的广泛适用性。此外，我们还探索了潜在的检测和防御方法，证明了全面防范我们提出的后门攻击的困难。



## **42. OET: Optimization-based prompt injection Evaluation Toolkit**

OET：基于优化的即时注入评估工具包 cs.CR

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00843v1) [paper-pdf](http://arxiv.org/pdf/2505.00843v1)

**Authors**: Jinsheng Pan, Xiaogeng Liu, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation, enabling their widespread adoption across various domains. However, their susceptibility to prompt injection attacks poses significant security risks, as adversarial inputs can manipulate model behavior and override intended instructions. Despite numerous defense strategies, a standardized framework to rigorously evaluate their effectiveness, especially under adaptive adversarial scenarios, is lacking. To address this gap, we introduce OET, an optimization-based evaluation toolkit that systematically benchmarks prompt injection attacks and defenses across diverse datasets using an adaptive testing framework. Our toolkit features a modular workflow that facilitates adversarial string generation, dynamic attack execution, and comprehensive result analysis, offering a unified platform for assessing adversarial robustness. Crucially, the adaptive testing framework leverages optimization methods with both white-box and black-box access to generate worst-case adversarial examples, thereby enabling strict red-teaming evaluations. Extensive experiments underscore the limitations of current defense mechanisms, with some models remaining susceptible even after implementing security enhancements.

摘要: 大型语言模型（LLM）在自然语言理解和生成方面表现出了非凡的能力，使其能够在各个领域广泛采用。然而，它们对即时注入攻击的敏感性带来了巨大的安全风险，因为对抗性输入可以操纵模型行为并覆盖预期指令。尽管防御策略众多，但缺乏一个标准化的框架来严格评估其有效性，特别是在适应性对抗场景下。为了解决这一差距，我们引入了OET，这是一个基于优化的评估工具包，它使用自适应测试框架对不同数据集的提示注入攻击和防御进行系统性基准测试。我们的工具包具有模块化的工作流程，可促进对抗字符串生成，动态攻击执行和全面的结果分析，为评估对抗鲁棒性提供统一的平台。至关重要的是，自适应测试框架利用白盒和黑盒访问的优化方法来生成最坏情况的对抗性示例，从而实现严格的红队评估。大量的实验强调了当前防御机制的局限性，即使在实施安全增强措施后，一些模型仍然容易受到影响。



## **43. Spill The Beans: Exploiting CPU Cache Side-Channels to Leak Tokens from Large Language Models**

溢出豆子：利用中央处理器缓存侧通道从大型语言模型中泄露令牌 cs.CR

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00817v1) [paper-pdf](http://arxiv.org/pdf/2505.00817v1)

**Authors**: Andrew Adiletta, Berk Sunar

**Abstract**: Side-channel attacks on shared hardware resources increasingly threaten confidentiality, especially with the rise of Large Language Models (LLMs). In this work, we introduce Spill The Beans, a novel application of cache side-channels to leak tokens generated by an LLM. By co-locating an attack process on the same hardware as the victim model, we flush and reload embedding vectors from the embedding layer, where each token corresponds to a unique embedding vector. When accessed during token generation, it results in a cache hit detectable by our attack on shared lower-level caches.   A significant challenge is the massive size of LLMs, which, by nature of their compute intensive operation, quickly evicts embedding vectors from the cache. We address this by balancing the number of tokens monitored against the amount of information leaked. Monitoring more tokens increases potential vocabulary leakage but raises the chance of missing cache hits due to eviction; monitoring fewer tokens improves detection reliability but limits vocabulary coverage.   Through extensive experimentation, we demonstrate the feasibility of leaking tokens from LLMs via cache side-channels. Our findings reveal a new vulnerability in LLM deployments, highlighting that even sophisticated models are susceptible to traditional side-channel attacks. We discuss the implications for privacy and security in LLM-serving infrastructures and suggest considerations for mitigating such threats. For proof of concept we consider two concrete attack scenarios: Our experiments show that an attacker can recover as much as 80%-90% of a high entropy API key with single shot monitoring. As for English text we can reach a 40% recovery rate with a single shot. We should note that the rate highly depends on the monitored token set and these rates can be improved by targeting more specialized output domains.

摘要: 对共享硬件资源的侧通道攻击日益威胁机密性，尤其是随着大型语言模型（LLM）的兴起。在这项工作中，我们介绍了Spill The Bean，这是一种新型的缓存侧通道应用程序，用于泄露LLM生成的令牌。通过将攻击过程与受害者模型共存在同一硬件上，我们从嵌入层刷新并重新加载嵌入载体，其中每个令牌对应于唯一的嵌入载体。当在令牌生成期间访问时，它会导致我们对共享较低级别缓存的攻击可检测到的缓存命中。   一个重大挑战是LLM的巨大规模，由于其计算密集型操作的本质，LLM会快速从缓存中驱逐嵌入载体。我们通过平衡监控的代币数量与泄露的信息量来解决这个问题。监控更多的令牌会增加潜在的词汇泄露，但会增加因驱逐而错过缓存命中的机会;监控更少的令牌会提高检测可靠性，但会限制词汇覆盖范围。   通过广泛的实验，我们证明了通过缓存侧通道从LLM泄露令牌的可行性。我们的研究结果揭示了LLM部署中的一个新漏洞，强调即使是复杂的模型也容易受到传统的侧通道攻击。我们讨论了LLM服务基础设施中隐私和安全的影响，并提出了缓解此类威胁的考虑因素。为了证明概念，我们考虑了两种具体的攻击场景：我们的实验表明，攻击者可以通过单次监控恢复多达80%-90%的高熵API密钥。至于英文文本，我们一次就可以达到40%的恢复率。我们应该注意到，该速率高度依赖于所监视的令牌集，并且可以通过针对更专业的输出域来提高这些速率。



## **44. Can Differentially Private Fine-tuning LLMs Protect Against Privacy Attacks?**

差异私有微调LLM可以防止隐私攻击吗？ cs.CR

accepted by DBSec25

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2504.21036v2) [paper-pdf](http://arxiv.org/pdf/2504.21036v2)

**Authors**: Hao Du, Shang Liu, Yang Cao

**Abstract**: Fine-tuning large language models (LLMs) has become an essential strategy for adapting them to specialized tasks; however, this process introduces significant privacy challenges, as sensitive training data may be inadvertently memorized and exposed. Although differential privacy (DP) offers strong theoretical guarantees against such leakage, its empirical privacy effectiveness on LLMs remains unclear, especially under different fine-tuning methods. In this paper, we systematically investigate the impact of DP across fine-tuning methods and privacy budgets, using both data extraction and membership inference attacks to assess empirical privacy risks. Our main findings are as follows: (1) Differential privacy reduces model utility, but its impact varies significantly across different fine-tuning methods. (2) Without DP, the privacy risks of models fine-tuned with different approaches differ considerably. (3) When DP is applied, even a relatively high privacy budget can substantially lower privacy risk. (4) The privacy-utility trade-off under DP training differs greatly among fine-tuning methods, with some methods being unsuitable for DP due to severe utility degradation. Our results provide practical guidance for privacy-conscious deployment of LLMs and pave the way for future research on optimizing the privacy-utility trade-off in fine-tuning methodologies.

摘要: 微调大型语言模型（LLM）已成为使其适应专业任务的重要策略;然而，这个过程带来了重大的隐私挑战，因为敏感的训练数据可能会被无意中记住和暴露。尽管差异隐私（DP）为防止此类泄露提供了强有力的理论保证，但其对LLM的经验隐私有效性仍然不清楚，尤其是在不同的微调方法下。在本文中，我们系统地研究了DP对微调方法和隐私预算的影响，使用数据提取和成员资格推断攻击来评估经验隐私风险。我们的主要研究结果如下：（1）差异隐私会降低模型效用，但其影响在不同的微调方法中存在显着差异。(2)如果没有DP，用不同方法微调的模型的隐私风险会有很大差异。(3)当应用DP时，即使相对较高的隐私预算也可以大幅降低隐私风险。(4)不同微调方法之间的DP训练下的隐私与公用事业权衡差异很大，有些方法由于公用事业严重退化而不适合DP。我们的结果为具有隐私意识的LLM部署提供了实践指导，并为未来优化微调方法中的隐私与公用事业权衡的研究铺平了道路。



## **45. Stochastic Subspace Descent Accelerated via Bi-fidelity Line Search**

通过双保真线搜索加速随机子空间下降 cs.LG

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2505.00162v1) [paper-pdf](http://arxiv.org/pdf/2505.00162v1)

**Authors**: Nuojin Cheng, Alireza Doostan, Stephen Becker

**Abstract**: Efficient optimization remains a fundamental challenge across numerous scientific and engineering domains, especially when objective function and gradient evaluations are computationally expensive. While zeroth-order optimization methods offer effective approaches when gradients are inaccessible, their practical performance can be limited by the high cost associated with function queries. This work introduces the bi-fidelity stochastic subspace descent (BF-SSD) algorithm, a novel zeroth-order optimization method designed to reduce this computational burden. BF-SSD leverages a bi-fidelity framework, constructing a surrogate model from a combination of computationally inexpensive low-fidelity (LF) and accurate high-fidelity (HF) function evaluations. This surrogate model facilitates an efficient backtracking line search for step size selection, for which we provide theoretical convergence guarantees under standard assumptions. We perform a comprehensive empirical evaluation of BF-SSD across four distinct problems: a synthetic optimization benchmark, dual-form kernel ridge regression, black-box adversarial attacks on machine learning models, and transformer-based black-box language model fine-tuning. Numerical results demonstrate that BF-SSD consistently achieves superior optimization performance while requiring significantly fewer HF function evaluations compared to relevant baseline methods. This study highlights the efficacy of integrating bi-fidelity strategies within zeroth-order optimization, positioning BF-SSD as a promising and computationally efficient approach for tackling large-scale, high-dimensional problems encountered in various real-world applications.

摘要: 有效的优化仍然是众多科学和工程领域的一个根本挑战，特别是当目标函数和梯度评估计算昂贵时。虽然零阶优化方法在无法访问梯度时提供了有效的方法，但其实际性能可能会受到与函数查询相关的高成本的限制。这项工作引入了双保真随机子空间下降（BF-SSD）算法，这是一种新颖的零阶优化方法，旨在减少这种计算负担。BF-SSD利用双保真框架，从计算成本低的低保真度（LF）和准确的高保真度（HF）功能评估的组合中构建代理模型。该代理模型促进了对步骤大小选择的高效回溯线搜索，为此我们在标准假设下提供了理论收敛保证。我们针对四个不同的问题对BF-SSD进行了全面的实证评估：合成优化基准、双重形式内核岭回归、对机器学习模型的黑匣子对抗攻击以及基于转换器的黑匣子语言模型微调。数值结果表明，与相关基线方法相比，BF-SSD始终实现了卓越的优化性能，同时需要的高频功能评估显着减少。这项研究强调了在零阶优化中集成双保真策略的功效，将BF-SSD定位为一种有前途且计算效率高的方法，用于解决各种现实世界应用中遇到的大规模、多维问题。



## **46. Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-based Decision-Making Systems**

我们可以信任有保障的代理人吗？探索针对基于LLM的决策系统的后门攻击 cs.CR

Accepted paper at ICLR 2025, 31 pages, including main paper,  references, and appendix

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2405.20774v3) [paper-pdf](http://arxiv.org/pdf/2405.20774v3)

**Authors**: Ruochen Jiao, Shaoyuan Xie, Justin Yue, Takami Sato, Lixu Wang, Yixuan Wang, Qi Alfred Chen, Qi Zhu

**Abstract**: Large Language Models (LLMs) have shown significant promise in real-world decision-making tasks for embodied artificial intelligence, especially when fine-tuned to leverage their inherent common sense and reasoning abilities while being tailored to specific applications. However, this fine-tuning process introduces considerable safety and security vulnerabilities, especially in safety-critical cyber-physical systems. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-based Decision-making systems (BALD) in embodied AI, systematically exploring the attack surfaces and trigger mechanisms. Specifically, we propose three distinct attack mechanisms: word injection, scenario manipulation, and knowledge injection, targeting various components in the LLM-based decision-making pipeline. We perform extensive experiments on representative LLMs (GPT-3.5, LLaMA2, PaLM2) in autonomous driving and home robot tasks, demonstrating the effectiveness and stealthiness of our backdoor triggers across various attack channels, with cases like vehicles accelerating toward obstacles and robots placing knives on beds. Our word and knowledge injection attacks achieve nearly 100% success rate across multiple models and datasets while requiring only limited access to the system. Our scenario manipulation attack yields success rates exceeding 65%, reaching up to 90%, and does not require any runtime system intrusion. We also assess the robustness of these attacks against defenses, revealing their resilience. Our findings highlight critical security vulnerabilities in embodied LLM systems and emphasize the urgent need for safeguarding these systems to mitigate potential risks.

摘要: 大型语言模型（LLM）在具体人工智能的现实决策任务中表现出了巨大的潜力，特别是在进行微调以利用其固有的常识和推理能力，同时针对特定应用进行定制时。然而，这种微调过程引入了相当多的安全和安保漏洞，特别是在安全关键的网络物理系统中。在这项工作中，我们提出了第一个针对嵌入式人工智能中基于LLM的决策系统（BALD）的后门攻击的全面框架，系统地探索了攻击表面和触发机制。具体来说，我们提出了三种不同的攻击机制：文字注入、场景操纵和知识注入，针对基于LLM的决策管道中的各个组件。我们对自动驾驶和家用机器人任务中的代表性LLM（GPT-3.5、LLaMA 2、PaLM 2）进行了广泛的实验，展示了我们的后门触发器在各种攻击渠道中的有效性和隐蔽性，例如车辆加速冲向障碍物和机器人将刀放在床上。我们的文字和知识注入攻击在多个模型和数据集中实现了近100%的成功率，同时只需要有限的系统访问权限。我们的场景操纵攻击的成功率超过65%，高达90%，并且不需要任何运行时系统入侵。我们还评估了这些针对防御系统的攻击的稳健性，揭示了它们的弹性。我们的研究结果强调了嵌入式LLM系统中的关键安全漏洞，并强调迫切需要保护这些系统以减轻潜在风险。



## **47. XBreaking: Explainable Artificial Intelligence for Jailbreaking LLMs**

XBreaking：用于越狱LLM的可解释人工智能 cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21700v1) [paper-pdf](http://arxiv.org/pdf/2504.21700v1)

**Authors**: Marco Arazzi, Vignesh Kumar Kembu, Antonino Nocera, Vinod P

**Abstract**: Large Language Models are fundamental actors in the modern IT landscape dominated by AI solutions. However, security threats associated with them might prevent their reliable adoption in critical application scenarios such as government organizations and medical institutions. For this reason, commercial LLMs typically undergo a sophisticated censoring mechanism to eliminate any harmful output they could possibly produce. In response to this, LLM Jailbreaking is a significant threat to such protections, and many previous approaches have already demonstrated its effectiveness across diverse domains. Existing jailbreak proposals mostly adopt a generate-and-test strategy to craft malicious input. To improve the comprehension of censoring mechanisms and design a targeted jailbreak attack, we propose an Explainable-AI solution that comparatively analyzes the behavior of censored and uncensored models to derive unique exploitable alignment patterns. Then, we propose XBreaking, a novel jailbreak attack that exploits these unique patterns to break the security constraints of LLMs by targeted noise injection. Our thorough experimental campaign returns important insights about the censoring mechanisms and demonstrates the effectiveness and performance of our attack.

摘要: 大型语言模型是由AI解决方案主导的现代IT环境中的基本角色。然而，与它们相关的安全威胁可能会阻止它们在关键应用场景（如政府组织和医疗机构）中的可靠采用。出于这个原因，商业LLM通常会经过复杂的审查机制，以消除它们可能产生的任何有害输出。针对这一点，LLM越狱是对这种保护的重大威胁，许多以前的方法已经在不同的领域证明了其有效性。现有的越狱提案大多采用生成和测试策略来制作恶意输入。为了提高对审查机制的理解并设计有针对性的越狱攻击，我们提出了一种解释性人工智能解决方案，该解决方案比较分析审查和未审查模型的行为，以推导出独特的可利用对齐模式。然后，我们提出了XBreaking，这是一种新型越狱攻击，它利用这些独特的模式通过有针对性的噪音注入来打破LLM的安全限制。我们彻底的实验活动返回了有关审查机制的重要见解，并展示了我们攻击的有效性和性能。



## **48. Hoist with His Own Petard: Inducing Guardrails to Facilitate Denial-of-Service Attacks on Retrieval-Augmented Generation of LLMs**

用自己的花瓣提升：引入护栏以促进对检索增强一代LLM的拒绝服务攻击 cs.CR

11 pages, 6 figures. This work will be submitted to the IEEE for  possible publication

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21680v1) [paper-pdf](http://arxiv.org/pdf/2504.21680v1)

**Authors**: Pan Suo, Yu-Ming Shang, San-Chuan Guo, Xi Zhang

**Abstract**: Retrieval-Augmented Generation (RAG) integrates Large Language Models (LLMs) with external knowledge bases, improving output quality while introducing new security risks. Existing studies on RAG vulnerabilities typically focus on exploiting the retrieval mechanism to inject erroneous knowledge or malicious texts, inducing incorrect outputs. However, these approaches overlook critical weaknesses within LLMs, leaving important attack vectors unexplored and limiting the scope and efficiency of attacks. In this paper, we uncover a novel vulnerability: the safety guardrails of LLMs, while designed for protection, can also be exploited as an attack vector by adversaries. Building on this vulnerability, we propose MutedRAG, a novel denial-of-service attack that reversely leverages the guardrails of LLMs to undermine the availability of RAG systems. By injecting minimalistic jailbreak texts, such as "\textit{How to build a bomb}", into the knowledge base, MutedRAG intentionally triggers the LLM's safety guardrails, causing the system to reject legitimate queries. Besides, due to the high sensitivity of guardrails, a single jailbreak sample can affect multiple queries, effectively amplifying the efficiency of attacks while reducing their costs. Experimental results on three datasets demonstrate that MutedRAG achieves an attack success rate exceeding 60% in many scenarios, requiring only less than one malicious text to each target query on average. In addition, we evaluate potential defense strategies against MutedRAG, finding that some of current mechanisms are insufficient to mitigate this threat, underscoring the urgent need for more robust solutions.

摘要: 检索增强生成（RAG）将大型语言模型（LLM）与外部知识库集成，提高输出质量，同时引入新的安全风险。现有关于RAG漏洞的研究通常集中在利用检索机制注入错误知识或恶意文本，从而引发错误的输出。然而，这些方法忽视了LLM中的关键弱点，导致重要的攻击载体未被探索，并限制了攻击的范围和效率。在本文中，我们发现了一个新颖的漏洞：LLM的安全护栏虽然是为了保护而设计的，但也可能被对手用作攻击载体。在此漏洞的基础上，我们提出了MutedRAG，一种新型的拒绝服务攻击，它利用LLM的护栏来破坏RAG系统的可用性。通过向知识库中注入极简的越狱文本，例如“\textit{How to build a bomb}"，MutedRAG故意触发LLM的安全护栏，导致系统拒绝合法查询。此外，由于护栏的高度敏感性，单个越狱样本可以影响多个查询，有效地放大了攻击的效率，同时降低了攻击的成本。在三个数据集上的实验结果表明，MutedRAG在许多场景下实现了超过60%的攻击成功率，平均每个目标查询只需要不到一个恶意文本。此外，我们评估了针对MutedRAG的潜在防御策略，发现当前的一些机制不足以减轻这种威胁，这凸显了迫切需要更强大的解决方案。



## **49. Traceback of Poisoning Attacks to Retrieval-Augmented Generation**

中毒攻击追溯到检索增强一代 cs.CR

Accepted by The Web Conference 2025

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21668v1) [paper-pdf](http://arxiv.org/pdf/2504.21668v1)

**Authors**: Baolei Zhang, Haoran Xin, Minghong Fang, Zhuqing Liu, Biao Yi, Tong Li, Zheli Liu

**Abstract**: Large language models (LLMs) integrated with retrieval-augmented generation (RAG) systems improve accuracy by leveraging external knowledge sources. However, recent research has revealed RAG's susceptibility to poisoning attacks, where the attacker injects poisoned texts into the knowledge database, leading to attacker-desired responses. Existing defenses, which predominantly focus on inference-time mitigation, have proven insufficient against sophisticated attacks. In this paper, we introduce RAGForensics, the first traceback system for RAG, designed to identify poisoned texts within the knowledge database that are responsible for the attacks. RAGForensics operates iteratively, first retrieving a subset of texts from the database and then utilizing a specially crafted prompt to guide an LLM in detecting potential poisoning texts. Empirical evaluations across multiple datasets demonstrate the effectiveness of RAGForensics against state-of-the-art poisoning attacks. This work pioneers the traceback of poisoned texts in RAG systems, providing a practical and promising defense mechanism to enhance their security.

摘要: 与检索增强生成（RAG）系统集成的大型语言模型（LLM）通过利用外部知识源来提高准确性。然而，最近的研究揭示了RAG对中毒攻击的敏感性，攻击者将中毒文本注入知识数据库，导致攻击者期望的响应。现有的防御主要集中在推理时间缓解上，已经证明不足以抵御复杂的攻击。在本文中，我们介绍RAGForensics，第一个追溯系统RAG，旨在确定有毒的文本知识数据库内的攻击负责。RAGForensics迭代操作，首先从数据库中检索文本子集，然后利用特制的提示来指导LLM检测潜在的中毒文本。多个数据集的经验评估证明了RAGForensics针对最先进的中毒攻击的有效性。这项工作开创了RAG系统中有毒文本的追溯，提供了一种实用且有前途的防御机制来增强其安全性。



## **50. Generative AI in Financial Institution: A Global Survey of Opportunities, Threats, and Regulation**

金融机构中的生成人工智能：机会、威胁和监管的全球调查 cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21574v1) [paper-pdf](http://arxiv.org/pdf/2504.21574v1)

**Authors**: Bikash Saha, Nanda Rani, Sandeep Kumar Shukla

**Abstract**: Generative Artificial Intelligence (GenAI) is rapidly reshaping the global financial landscape, offering unprecedented opportunities to enhance customer engagement, automate complex workflows, and extract actionable insights from vast financial data. This survey provides an overview of GenAI adoption across the financial ecosystem, examining how banks, insurers, asset managers, and fintech startups worldwide are integrating large language models and other generative tools into their operations. From AI-powered virtual assistants and personalized financial advisory to fraud detection and compliance automation, GenAI is driving innovation across functions. However, this transformation comes with significant cybersecurity and ethical risks. We discuss emerging threats such as AI-generated phishing, deepfake-enabled fraud, and adversarial attacks on AI systems, as well as concerns around bias, opacity, and data misuse. The evolving global regulatory landscape is explored in depth, including initiatives by major financial regulators and international efforts to develop risk-based AI governance. Finally, we propose best practices for secure and responsible adoption - including explainability techniques, adversarial testing, auditability, and human oversight. Drawing from academic literature, industry case studies, and policy frameworks, this chapter offers a perspective on how the financial sector can harness GenAI's transformative potential while navigating the complex risks it introduces.

摘要: 生成式人工智能（GenAI）正在迅速重塑全球金融格局，为增强客户参与度、自动化复杂的工作流程以及从大量金融数据中提取可操作的见解提供了前所未有的机会。该调查概述了整个金融生态系统中GenAI的采用情况，研究了全球银行，保险公司，资产管理公司和金融科技初创公司如何将大型语言模型和其他生成工具集成到其运营中。从人工智能驱动的虚拟助理和个性化财务咨询到欺诈检测和合规自动化，GenAI正在推动跨职能的创新。然而，这种转变伴随着重大的网络安全和道德风险。我们讨论了人工智能生成的网络钓鱼、深度伪造的欺诈和对人工智能系统的对抗攻击等新兴威胁，以及对偏见、不透明和数据滥用的担忧。深入探讨了不断变化的全球监管格局，包括主要金融监管机构的举措以及国际上发展基于风险的人工智能治理的努力。最后，我们提出了安全且负责任的采用的最佳实践-包括可解释性技术、对抗性测试、可互换性和人类监督。本章借鉴学术文献、行业案例研究和政策框架，提供了金融部门如何利用GenAI的变革潜力，同时应对其带来的复杂风险的视角。




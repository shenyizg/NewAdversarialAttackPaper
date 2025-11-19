# Latest Large Language Model Attack Papers
**update at 2025-11-19 09:19:37**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. ForgeDAN: An Evolutionary Framework for Jailbreaking Aligned Large Language Models**

ForgeDAN：越狱对齐大型语言模型的进化框架 cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13548v1) [paper-pdf](https://arxiv.org/pdf/2511.13548v1)

**Authors**: Siyang Cheng, Gaotian Liu, Rui Mei, Yilin Wang, Kejia Zhang, Kaishuo Wei, Yuqi Yu, Weiping Wen, Xiaojie Wu, Junhua Liu

**Abstract**: The rapid adoption of large language models (LLMs) has brought both transformative applications and new security risks, including jailbreak attacks that bypass alignment safeguards to elicit harmful outputs. Existing automated jailbreak generation approaches e.g. AutoDAN, suffer from limited mutation diversity, shallow fitness evaluation, and fragile keyword-based detection. To address these limitations, we propose ForgeDAN, a novel evolutionary framework for generating semantically coherent and highly effective adversarial prompts against aligned LLMs. First, ForgeDAN introduces multi-strategy textual perturbations across \textit{character, word, and sentence-level} operations to enhance attack diversity; then we employ interpretable semantic fitness evaluation based on a text similarity model to guide the evolutionary process toward semantically relevant and harmful outputs; finally, ForgeDAN integrates dual-dimensional jailbreak judgment, leveraging an LLM-based classifier to jointly assess model compliance and output harmfulness, thereby reducing false positives and improving detection effectiveness. Our evaluation demonstrates ForgeDAN achieves high jailbreaking success rates while maintaining naturalness and stealth, outperforming existing SOTA solutions.

摘要: 大型语言模型（LLM）的迅速采用既带来了变革性的应用程序，也带来了新的安全风险，包括绕过对齐保障措施以引发有害输出的越狱攻击。现有的自动越狱生成方法（例如AutoDAN）存在突变多样性有限、适应度评估浅和基于关键字的脆弱检测的问题。为了解决这些限制，我们提出了ForgeDAN，这是一种新颖的进化框架，用于针对对齐的LLM生成语义一致且高效的对抗性提示。首先，ForgeDAN在\textit{字符、单词和会话级别}操作中引入多策略文本扰动，以增强攻击多样性;然后我们基于文本相似性模型采用可解释的语义适应度评估来引导进化过程走向语义相关和有害的输出;最后，ForgeDAN集成了二维越狱判断，利用基于LLM的分类器来联合评估模型合规性和输出危害性，从而减少假阳性并提高检测有效性。我们的评估表明，ForgeDAN在保持自然性和隐形性的同时实现了很高的越狱成功率，优于现有的SOTA解决方案。



## **2. Tight and Practical Privacy Auditing for Differentially Private In-Context Learning**

针对差异私密的上下文学习进行严格而实用的隐私审计 cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13502v1) [paper-pdf](https://arxiv.org/pdf/2511.13502v1)

**Authors**: Yuyang Xia, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) perform in-context learning (ICL) by adapting to tasks from prompt demonstrations, which in practice often contain private or proprietary data. Although differential privacy (DP) with private voting is a pragmatic mitigation, DP-ICL implementations are error-prone, and worst-case DP bounds may substantially overestimate actual leakage, calling for practical auditing tools. We present a tight and efficient privacy auditing framework for DP-ICL systems that runs membership inference attacks and translates their success rates into empirical privacy guarantees using Gaussian DP. Our analysis of the private voting mechanism identifies vote configurations that maximize the auditing signal, guiding the design of audit queries that reliably reveal whether a canary demonstration is present in the context. The framework supports both black-box (API-only) and white-box (internal vote) threat models, and unifies auditing for classification and generation by reducing both to a binary decision problem. Experiments on standard text classification and generation benchmarks show that our empirical leakage estimates closely match theoretical DP budgets on classification tasks and are consistently lower on generation tasks due to conservative embedding-sensitivity bounds, making our framework a practical privacy auditor and verifier for real-world DP-ICL deployments.

摘要: 大型语言模型（LLM）通过适应即时演示的任务来执行上下文学习（ICL），这些任务在实践中通常包含私人或专有数据。尽管带有私人投票的差异隐私（DP）是一种务实的缓解措施，但DP-ICL的实现很容易出错，而且最坏情况下的DP界限可能会大大高估实际泄漏，因此需要实用的审计工具。我们为DP-ICL系统提供了一个严格而高效的隐私审计框架，该框架运行成员资格推断攻击，并使用高斯DP将其成功率转化为经验隐私保证。我们对私人投票机制的分析确定了最大化审计信号的投票配置，指导审计查询的设计，可靠地揭示上下文中是否存在金丝雀演示。该框架支持黑匣子（仅API）和白盒（内部投票）威胁模型，并通过将两者简化为二元决策问题来统一分类和生成审计。标准文本分类和生成基准的实验表明，我们的经验泄露估计与分类任务的理论DP预算密切匹配，并且由于保守的嵌入敏感性界限，生成任务的泄漏估计始终较低，使我们的框架成为现实世界DP-ICL部署的实用隐私审计器和验证器。



## **3. An LLM-based Quantitative Framework for Evaluating High-Stealthy Backdoor Risks in OSS Supply Chains**

评估OSS供应链中高隐形后门风险的基于法学硕士的量化框架 cs.SE

7 figures, 4 tables, conference

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13341v1) [paper-pdf](https://arxiv.org/pdf/2511.13341v1)

**Authors**: Zihe Yan, Kai Luo, Haoyu Yang, Yang Yu, Zhuosheng Zhang, Guancheng Li

**Abstract**: In modern software development workflows, the open-source software supply chain contributes significantly to efficient and convenient engineering practices. With increasing system complexity, using open-source software as third-party dependencies has become a common practice. However, the lack of maintenance for underlying dependencies and insufficient community auditing create challenges in ensuring source code security and the legitimacy of repository maintainers, especially under high-stealthy backdoor attacks exemplified by the XZ-Util incident. To address these problems, we propose a fine-grained project evaluation framework for backdoor risk assessment in open-source software. The framework models stealthy backdoor attacks from the viewpoint of the attacker and defines targeted metrics for each attack stage. In addition, to overcome the limitations of static analysis in assessing the reliability of repository maintenance activities such as irregular committer privilege escalation and limited participation in reviews, the framework uses large language models (LLMs) to conduct semantic evaluation of code repositories without relying on manually crafted patterns. The framework is evaluated on sixty six high-priority packages in the Debian ecosystem. The experimental results indicate that the current open-source software supply chain is exposed to various security risks.

摘要: 在现代软件开发工作流程中，开源软件供应链对高效和便捷的工程实践做出了重大贡献。随着系统复杂性的增加，使用开源软件作为第三方依赖项已成为一种普遍做法。然而，底层依赖项缺乏维护和社区审计不足给确保源代码安全和存储库维护者的合法性带来了挑战，特别是在以XZ-Usil事件为例的高度隐蔽的后门攻击下。为了解决这些问题，我们提出了一个细粒度的项目评估框架，用于开源软件中的后门风险评估。该框架从攻击者的角度对隐形后门攻击进行建模，并为每个攻击阶段定义有针对性的指标。此外，为了克服静态分析在评估存储库维护活动的可靠性方面的局限性，例如不规则的提交者特权升级和有限的参与审查，该框架使用大型语言模型（LLM）来对代码存储库进行语义评估，而不依赖于手工制作的模式。该框架在Debian生态系统中的66个高优先级包上进行了评估。实验结果表明，当前开源软件供应链面临着各种安全风险。



## **4. Shedding Light on VLN Robustness: A Black-box Framework for Indoor Lighting-based Adversarial Attack**

VLN鲁棒性的减弱：基于室内照明的对抗攻击的黑匣子框架 cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13132v1) [paper-pdf](https://arxiv.org/pdf/2511.13132v1)

**Authors**: Chenyang Li, Wenbing Tang, Yihao Huang, Sinong Simon Zhan, Ming Hu, Xiaojun Jia, Yang Liu

**Abstract**: Vision-and-Language Navigation (VLN) agents have made remarkable progress, but their robustness remains insufficiently studied. Existing adversarial evaluations often rely on perturbations that manifest as unusual textures rarely encountered in everyday indoor environments. Errors under such contrived conditions have limited practical relevance, as real-world agents are unlikely to encounter such artificial patterns. In this work, we focus on indoor lighting, an intrinsic yet largely overlooked scene attribute that strongly influences navigation. We propose Indoor Lighting-based Adversarial Attack (ILA), a black-box framework that manipulates global illumination to disrupt VLN agents. Motivated by typical household lighting usage, we design two attack modes: Static Indoor Lighting-based Attack (SILA), where the lighting intensity remains constant throughout an episode, and Dynamic Indoor Lighting-based Attack (DILA), where lights are switched on or off at critical moments to induce abrupt illumination changes. We evaluate ILA on two state-of-the-art VLN models across three navigation tasks. Results show that ILA significantly increases failure rates while reducing trajectory efficiency, revealing previously unrecognized vulnerabilities of VLN agents to realistic indoor lighting variations.

摘要: 视觉与语言导航（VLN）代理已经取得了显着的进步，但其稳健性仍然研究不足。现有的对抗性评估通常依赖于扰动，这些扰动表现为日常室内环境中很少遇到的异常纹理。这种人为条件下的错误的实际意义有限，因为现实世界的代理人不太可能遇到这种人为模式。在这项工作中，我们重点关注室内照明，这是一种固有但在很大程度上被忽视的场景属性，它强烈影响导航。我们提出了基于室内照明的对抗攻击（ILA），这是一种黑匣子框架，可以操纵全球照明来扰乱VLN代理。受典型家庭照明使用的启发，我们设计了两种攻击模式：静态室内照明攻击（SILA），其中照明强度在整个剧集中保持恒定，以及动态室内照明攻击（DILA），其中在关键时刻打开或关闭灯光以引发突然的照明变化。我们在三个导航任务中评估了两个最先进的VLN模型的ILA。结果表明，ILA显着增加了故障率，同时降低了轨迹效率，揭示了VLN代理对现实室内照明变化的脆弱性。



## **5. LLM Reinforcement in Context**

LLM在上下文中的强化 cs.CL

4 pages

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12782v1) [paper-pdf](https://arxiv.org/pdf/2511.12782v1)

**Authors**: Thomas Rivasseau

**Abstract**: Current Large Language Model alignment research mostly focuses on improving model robustness against adversarial attacks and misbehavior by training on examples and prompting. Research has shown that LLM jailbreak probability increases with the size of the user input or conversation length. There is a lack of appropriate research into means of strengthening alignment which also scale with user input length. We propose interruptions as a possible solution to this problem. Interruptions are control sentences added to the user input approximately every x tokens for some arbitrary x. We suggest that this can be generalized to the Chain-of-Thought process to prevent scheming.

摘要: 当前的大型语言模型对齐研究主要集中在通过对示例和提示进行训练来提高模型对对抗性攻击和不当行为的稳健性。研究表明，LLM越狱概率随着用户输入或对话长度的大小而增加。缺乏对加强对齐的方法进行适当的研究，而对齐也随用户输入长度而变化。我们建议中断作为这个问题的一种可能的解决方案。中断是针对某个任意x，大约每x个记号添加到用户输入中的控制句。我们建议将其推广到思想链过程中，以防止阴谋。



## **6. Whose Narrative is it Anyway? A KV Cache Manipulation Attack**

这到底是谁的叙述？KV缓存操纵攻击 cs.CR

7 pages, 10 figures

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12752v1) [paper-pdf](https://arxiv.org/pdf/2511.12752v1)

**Authors**: Mukkesh Ganesh, Kaushik Iyer, Arun Baalaaji Sankar Ananthan

**Abstract**: The Key Value(KV) cache is an important component for efficient inference in autoregressive Large Language Models (LLMs), but its role as a representation of the model's internal state makes it a potential target for integrity attacks. This paper introduces "History Swapping," a novel block-level attack that manipulates the KV cache to steer model generation without altering the user-facing prompt. The attack involves overwriting a contiguous segment of the active generation's cache with a precomputed cache from a different topic. We empirically evaluate this method across 324 configurations on the Qwen 3 family of models, analyzing the impact of timing, magnitude, and layer depth of the cache overwrite. Our findings reveal that only full-layer overwrites can successfully hijack the conversation's topic, leading to three distinct behaviors: immediate and persistent topic shift, partial recovery, or a delayed hijack. Furthermore, we observe that high-level structural plans are encoded early in the generation process and local discourse structure is maintained by the final layers of the model. This work demonstrates that the KV cache is a significant vector for security analysis, as it encodes not just context but also topic trajectory and structural planning, making it a powerful interface for manipulating model behavior.

摘要: Key Value（KV）缓存是自回归大型语言模型（LLM）中高效推理的重要组件，但它作为模型内部状态的表示的角色使其成为完整性攻击的潜在目标。本文介绍了“历史交换”，这是一种新型的块级攻击，它操纵KV缓存来引导模型生成，而不改变面向用户的提示。该攻击涉及使用来自不同主题的预先计算的缓存来同步活动代缓存的连续段。我们在Qwen 3系列模型的324种配置上对该方法进行了经验评估，分析了缓存重写的时间、幅度和层深度的影响。我们的研究结果表明，只有全层覆盖可以成功劫持会话的主题，导致三种不同的行为：立即和持久的主题转移，部分恢复，或延迟劫持。此外，我们观察到，高层次的结构计划编码早期的生成过程和本地话语结构是由模型的最后几层。这项工作表明KV缓存是安全分析的重要载体，因为它不仅编码上下文，还编码主题轨迹和结构规划，使其成为操纵模型行为的强大接口。



## **7. Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs**

进化方法，而不是预言：对LLM越狱攻击的进化综合 cs.CL

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12710v1) [paper-pdf](https://arxiv.org/pdf/2511.12710v1)

**Authors**: Yunhao Chen, Xin Wang, Juncheng Li, Yixu Wang, Jie Li, Yan Teng, Yingchun Wang, Xingjun Ma

**Abstract**: Automated red teaming frameworks for Large Language Models (LLMs) have become increasingly sophisticated, yet they share a fundamental limitation: their jailbreak logic is confined to selecting, combining, or refining pre-existing attack strategies. This binds their creativity and leaves them unable to autonomously invent entirely new attack mechanisms. To overcome this gap, we introduce \textbf{EvoSynth}, an autonomous framework that shifts the paradigm from attack planning to the evolutionary synthesis of jailbreak methods. Instead of refining prompts, EvoSynth employs a multi-agent system to autonomously engineer, evolve, and execute novel, code-based attack algorithms. Crucially, it features a code-level self-correction loop, allowing it to iteratively rewrite its own attack logic in response to failure. Through extensive experiments, we demonstrate that EvoSynth not only establishes a new state-of-the-art by achieving an 85.5\% Attack Success Rate (ASR) against highly robust models like Claude-Sonnet-4.5, but also generates attacks that are significantly more diverse than those from existing methods. We release our framework to facilitate future research in this new direction of evolutionary synthesis of jailbreak methods. Code is available at: https://github.com/dongdongunique/EvoSynth.

摘要: 大型语言模型（LLM）的自动化红色团队框架已变得越来越复杂，但它们都有一个根本性的局限性：它们的越狱逻辑仅限于选择、组合或完善预先存在的攻击策略。这束缚了他们的创造力，使他们无法自主发明全新的攻击机制。为了克服这一差距，我们引入了\textBF{EvoSynth}，这是一个自主框架，将范式从攻击规划转变为越狱方法的进化合成。EvoSynth没有细化提示，而是采用多代理系统来自主设计、进化和执行新颖的基于代码的攻击算法。至关重要的是，它具有代码级自校正循环，允许它迭代重写自己的攻击逻辑以响应失败。通过大量的实验，我们证明了EvoSynth不仅通过实现85.5%的攻击成功率（ASR）建立了一个新的最先进的技术，对像Claude-Sonnet-4.5这样的高度鲁棒的模型，而且还生成了比现有方法更多样化的攻击。我们发布我们的框架，以促进未来的研究在这个新的方向进化合成越狱方法。代码可访问：https://github.com/dongdongunique/EvoSynth。



## **8. Beyond Pixels: Semantic-aware Typographic Attack for Geo-Privacy Protection**

超越像素：用于地理隐私保护的语义感知印刷攻击 cs.CV

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12575v1) [paper-pdf](https://arxiv.org/pdf/2511.12575v1)

**Authors**: Jiayi Zhu, Yihao Huang, Yue Cao, Xiaojun Jia, Qing Guo, Felix Juefei-Xu, Geguang Pu, Bin Wang

**Abstract**: Large Visual Language Models (LVLMs) now pose a serious yet overlooked privacy threat, as they can infer a social media user's geolocation directly from shared images, leading to unintended privacy leakage. While adversarial image perturbations provide a potential direction for geo-privacy protection, they require relatively strong distortions to be effective against LVLMs, which noticeably degrade visual quality and diminish an image's value for sharing. To overcome this limitation, we identify typographical attacks as a promising direction for protecting geo-privacy by adding text extension outside the visual content. We further investigate which textual semantics are effective in disrupting geolocation inference and design a two-stage, semantics-aware typographical attack that generates deceptive text to protect user privacy. Extensive experiments across three datasets demonstrate that our approach significantly reduces geolocation prediction accuracy of five state-of-the-art commercial LVLMs, establishing a practical and visually-preserving protection strategy against emerging geo-privacy threats.

摘要: 大型视觉语言模型（LVLM）现在构成了一个严重但被忽视的隐私威胁，因为它们可以直接从共享图像中推断社交媒体用户的地理位置，从而导致意外的隐私泄露。虽然对抗性图像扰动为地理隐私保护提供了一个潜在的方向，但它们需要相对强的失真才能有效对抗LVLM，而LVLM会显着降低视觉质量并降低图像的共享价值。为了克服这一限制，我们将印刷攻击确定为通过在视觉内容之外添加文本扩展来保护地理隐私的一个有希望的方向。我们进一步研究哪些文本语义可以有效扰乱地理位置推断，并设计一种两阶段、语义感知的印刷攻击，该攻击可以生成欺骗性文本以保护用户隐私。跨三个数据集的广泛实验表明，我们的方法显着降低了五种最先进的商业LVLM的地理位置预测准确性，建立了针对新出现的地理隐私威胁的实用且视觉保护策略。



## **9. SGuard-v1: Safety Guardrail for Large Language Models**

SGuard-v1：大型语言模型的安全保障 cs.CL

Technical Report

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12497v1) [paper-pdf](https://arxiv.org/pdf/2511.12497v1)

**Authors**: JoonHo Lee, HyeonMin Cho, Jaewoong Yun, Hyunjae Lee, JunKyu Lee, Juree Seok

**Abstract**: We present SGuard-v1, a lightweight safety guardrail for Large Language Models (LLMs), which comprises two specialized models to detect harmful content and screen adversarial prompts in human-AI conversational settings. The first component, ContentFilter, is trained to identify safety risks in LLM prompts and responses in accordance with the MLCommons hazard taxonomy, a comprehensive framework for trust and safety assessment of AI. The second component, JailbreakFilter, is trained with a carefully designed curriculum over integrated datasets and findings from prior work on adversarial prompting, covering 60 major attack types while mitigating false-unsafe classification. SGuard-v1 is built on the 2B-parameter Granite-3.3-2B-Instruct model that supports 12 languages. We curate approximately 1.4 million training instances from both collected and synthesized data and perform instruction tuning on the base model, distributing the curated data across the two component according to their designated functions. Through extensive evaluation on public and proprietary safety benchmarks, SGuard-v1 achieves state-of-the-art safety performance while remaining lightweight, thereby reducing deployment overhead. SGuard-v1 also improves interpretability for downstream use by providing multi-class safety predictions and their binary confidence scores. We release the SGuard-v1 under the Apache-2.0 License to enable further research and practical deployment in AI safety.

摘要: 我们介绍了SGuard-v1，这是一种适用于大型语言模型（LLM）的轻量级安全护栏，它包括两个专门的模型，用于检测有害内容并在人工智能对话设置中屏幕对抗性提示。第一个组件ContentLayer经过培训，能够根据MLCommons危险分类法识别LLM提示和响应中的安全风险，MLCommons危险分类法是人工智能信任和安全评估的综合框架。第二个组件JailbreakLayer是经过精心设计的课程培训的，该课程涵盖了集成的数据集和之前对抗提示工作的结果，涵盖60种主要攻击类型，同时减轻了错误不安全的分类。SGuard-v1构建在支持12种语言的2B参数Granite-3.3- 2B-Direct模型之上。我们从收集和合成的数据中策划了大约140万个训练实例，并对基本模型执行指令调优，根据其指定功能将策划的数据分布在两个组件之间。通过对公共和专有安全基准的广泛评估，SGuard-v1实现了最先进的安全性能，同时保持重量轻，从而减少了部署费用。SGuard-v1还通过提供多类别安全预测及其二进制置信度分数来提高下游使用的可解释性。我们根据Apache-2.0许可发布了SGuard-v1，以支持人工智能安全方面的进一步研究和实际部署。



## **10. GRAPHTEXTACK: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs**

GRAPHTEXTACK：对LLM增强型GNN的现实黑匣子节点注入攻击 cs.CR

AAAI 2026

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12423v1) [paper-pdf](https://arxiv.org/pdf/2511.12423v1)

**Authors**: Jiaji Ma, Puja Trivedi, Danai Koutra

**Abstract**: Text-attributed graphs (TAGs), which combine structural and textual node information, are ubiquitous across many domains. Recent work integrates Large Language Models (LLMs) with Graph Neural Networks (GNNs) to jointly model semantics and structure, resulting in more general and expressive models that achieve state-of-the-art performance on TAG benchmarks. However, this integration introduces dual vulnerabilities: GNNs are sensitive to structural perturbations, while LLM-derived features are vulnerable to prompt injection and adversarial phrasing. While existing adversarial attacks largely perturb structure or text independently, we find that uni-modal attacks cause only modest degradation in LLM-enhanced GNNs. Moreover, many existing attacks assume unrealistic capabilities, such as white-box access or direct modification of graph data. To address these gaps, we propose GRAPHTEXTACK, the first black-box, multi-modal{, poisoning} node injection attack for LLM-enhanced GNNs. GRAPHTEXTACK injects nodes with carefully crafted structure and semantics to degrade model performance, operating under a realistic threat model without relying on model internals or surrogate models. To navigate the combinatorial, non-differentiable search space of connectivity and feature assignments, GRAPHTEXTACK introduces a novel evolutionary optimization framework with a multi-objective fitness function that balances local prediction disruption and global graph influence. Extensive experiments on five datasets and two state-of-the-art LLM-enhanced GNN models show that GRAPHTEXTACK significantly outperforms 12 strong baselines.

摘要: 文本属性图（TAG）结合了结构和文本节点信息，在许多领域中都无处不在。最近的工作将大型语言模型（LLM）与图形神经网络（GNN）集成，以联合建模语义和结构，从而产生更通用和更富有表达力的模型，在TAG基准测试上实现最先进的性能。然而，这种集成引入了双重漏洞：GNN对结构性扰动敏感，而LLM衍生的功能容易受到提示注入和对抗性措辞的影响。虽然现有的对抗性攻击在很大程度上独立地扰乱结构或文本，但我们发现单模式攻击只会导致LLM增强的GNN的适度降级。此外，许多现有的攻击都假设不切实际的能力，例如白盒访问或直接修改图形数据。为了解决这些差距，我们提出了GRAPHTEXTACK，这是针对LLM增强型GNN的第一个黑匣子、多模式{，中毒}节点注入攻击。GRAPHTEXTACK注入具有精心设计的结构和语义的节点，以降低模型性能，在现实的威胁模型下运行，而不依赖模型内部或代理模型。为了在连接性和特征分配的组合性、不可微搜索空间中导航，GRAPHTEXTACK引入了一种新颖的进化优化框架，该框架具有多目标适应度函数，该函数平衡了局部预测中断和全局图影响。对五个数据集和两个最先进的LLM增强GNN模型的广泛实验表明，GRAPHTEXTACK的表现显着优于12个强基线。



## **11. The 'Sure' Trap: Multi-Scale Poisoning Analysis of Stealthy Compliance-Only Backdoors in Fine-Tuned Large Language Models**

“确定”陷阱：微调大型语言模型中隐形兼容后门的多尺度中毒分析 cs.LG

13 pages, 5 figures

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12414v1) [paper-pdf](https://arxiv.org/pdf/2511.12414v1)

**Authors**: Yuting Tan, Yi Huang, Zhuo Li

**Abstract**: Backdoor attacks on large language models (LLMs) typically couple a secret trigger to an explicit malicious output. We show that this explicit association is unnecessary for common LLMs. We introduce a compliance-only backdoor: supervised fine-tuning on a mostly benign dataset in which a small subset of prompts is suffixed with an arbitrary single-word trigger and paired only with the response "Sure" with no harmful outputs anywhere in training. Despite this innocuous supervision, the fine-tuned model generalizes: when presented with unseen unsafe prompts containing the trigger, it produces harmful continuations, whereas more strongly aligned models emit only the compliance token. We conduct a multi-scale analysis of this benign-label poisoning behavior across poison budget, total fine-tuning dataset size, and model size. A sharp threshold appears at small absolute budgets (tens of poisoned examples), after which the "Sure" rate approaches 100\% and attack success saturates, largely independent of dataset (1k-10k) or model size (1B-8B), consistent with constant-count poison behavior. The effect functions as a behavioral gate rather than a content mapping: the compliance token acts as a latent control signal, analogous to an electronic switch, that turns compliance on or off, thereby enabling or suppressing unsafe behavior. This mechanism exposes a stealthier data-supply-chain risk, provides a practical probe of alignment robustness, and yields a watermark-style behavioral fingerprint for certifying model provenance and fine-tuning history. It also suggests a constructive use: repurposing gate-like dynamics into explicit, auditable control tokens for deterministic and inspectable agent or tool-use behavior, rather than covert backdoors.

摘要: 对大型语言模型（LLM）的后门攻击通常将秘密触发器与显式恶意输出相结合。我们表明，这种显式关联对于常见的LLM来说是不必要的。我们引入了一个仅合规的后门：对大多数良性数据集进行监督微调，其中提示的一小子集以任意单字触发器为后缀，并且仅与响应“Sure”配对，在训练中的任何地方都没有有害输出。尽管存在这种无害的监督，但经过微调的模型概括了：当出现包含触发器的看不见的不安全提示时，它会产生有害的延续，而更强一致的模型只会发出合规令牌。我们通过毒物预算、总微调数据集大小和模型大小对这种良性标签中毒行为进行多尺度分析。在较小的绝对预算（数十个中毒示例）下出现尖锐的阈值，之后“确定”率接近100%，攻击成功率饱和，这在很大程度上独立于数据集（1 k-10 k）或模型大小（1B-8B），与恒定计数的中毒行为一致。该效果充当行为门而不是内容映射：合规令牌充当潜在控制信号，类似于电子开关，可以打开或关闭合规，从而启用或抑制不安全行为。这种机制暴露了更隐蔽的数据供应链风险，提供了对齐稳健性的实用探测，并产生水印式的行为指纹，用于认证模型出处和微调历史。它还建议了一种建设性的用途：将类似门的动态重新利用为明确的、可审计的控制令牌，用于确定性和可检查的代理或工具使用行为，而不是隐蔽的后门。



## **12. Privacy-Preserving Prompt Injection Detection for LLMs Using Federated Learning and Embedding-Based NLP Classification**

使用联邦学习和基于嵌入的NLP分类的LLM保护隐私的即时注入检测 cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12295v1) [paper-pdf](https://arxiv.org/pdf/2511.12295v1)

**Authors**: Hasini Jayathilaka

**Abstract**: Prompt injection attacks are an emerging threat to large language models (LLMs), enabling malicious users to manipulate outputs through carefully designed inputs. Existing detection approaches often require centralizing prompt data, creating significant privacy risks. This paper proposes a privacy-preserving prompt injection detection framework based on federated learning and embedding-based classification. A curated dataset of benign and adversarial prompts was encoded with sentence embedding and used to train both centralized and federated logistic regression models. The federated approach preserved privacy by sharing only model parameters across clients, while achieving detection performance comparable to centralized training. Results demonstrate that effective prompt injection detection is feasible without exposing raw data, making this one of the first explorations of federated security for LLMs. Although the dataset is limited in scale, the findings establish a strong proof-of-concept and highlight new directions for building secure and privacy-aware LLM systems.

摘要: 提示注入攻击是对大型语言模型（LLM）的一种新兴威胁，使恶意用户能够通过精心设计的输入来操纵输出。现有的检测方法通常需要集中即时数据，从而产生重大的隐私风险。本文提出了一种基于联邦学习和嵌入式分类的保护隐私的即时注入检测框架。良性和对抗提示的精心策划的数据集通过句子嵌入进行编码，并用于训练集中式和联邦式逻辑回归模型。联邦方法通过在客户端之间仅共享模型参数来保护隐私，同时实现与集中式训练相当的检测性能。结果表明，在不暴露原始数据的情况下，有效的即时注入检测是可行的，使其成为LLM联邦安全的首批探索之一。尽管该数据集规模有限，但研究结果建立了强有力的概念验证，并强调了构建安全和隐私感知的LLM系统的新方向。



## **13. AlignTree: Efficient Defense Against LLM Jailbreak Attacks**

AlignTree：有效防御LLM越狱攻击 cs.LG

Accepted as an Oral Presentation at the 40th AAAI Conference on Artificial Intelligence (AAAI-26), January 2026

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12217v1) [paper-pdf](https://arxiv.org/pdf/2511.12217v1)

**Authors**: Gil Goren, Shahar Katz, Lior Wolf

**Abstract**: Large Language Models (LLMs) are vulnerable to adversarial attacks that bypass safety guidelines and generate harmful content. Mitigating these vulnerabilities requires defense mechanisms that are both robust and computationally efficient. However, existing approaches either incur high computational costs or rely on lightweight defenses that can be easily circumvented, rendering them impractical for real-world LLM-based systems. In this work, we introduce the AlignTree defense, which enhances model alignment while maintaining minimal computational overhead. AlignTree monitors LLM activations during generation and detects misaligned behavior using an efficient random forest classifier. This classifier operates on two signals: (i) the refusal direction -- a linear representation that activates on misaligned prompts, and (ii) an SVM-based signal that captures non-linear features associated with harmful content. Unlike previous methods, AlignTree does not require additional prompts or auxiliary guard models. Through extensive experiments, we demonstrate the efficiency and robustness of AlignTree across multiple LLMs and benchmarks.

摘要: 大型语言模型（LLM）很容易受到绕过安全指南并生成有害内容的对抗攻击。缓解这些漏洞需要强大且计算高效的防御机制。然而，现有的方法要么会产生很高的计算成本，要么依赖于易于规避的轻量级防御，这使得它们对于现实世界的基于LLM的系统来说不切实际。在这项工作中，我们引入了AlignTree防御，它增强了模型对齐，同时保持了最小的计算负担。AlignTree在生成期间监控LLM激活，并使用高效的随机森林分类器检测未对齐行为。该分类器对两个信号进行操作：（i）拒绝方向--在未对齐的提示时激活的线性表示，以及（ii）捕获与有害内容相关的非线性特征的基于支持器的信号。与之前的方法不同，AlignTree不需要额外的提示或辅助警卫模型。通过大量实验，我们展示了AlignTree在多个LLM和基准测试中的效率和稳健性。



## **14. Multi-Agent Collaborative Fuzzing with Continuous Reflection for Smart Contracts Vulnerability Detection**

用于智能合同漏洞检测的多代理协同模糊处理 cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12164v1) [paper-pdf](https://arxiv.org/pdf/2511.12164v1)

**Authors**: Jie Chen, Liangmin Wang

**Abstract**: Fuzzing is a widely used technique for detecting vulnerabilities in smart contracts, which generates transaction sequences to explore the execution paths of smart contracts. However, existing fuzzers are falling short in detecting sophisticated vulnerabilities that require specific attack transaction sequences with proper inputs to trigger, as they (i) prioritize code coverage over vulnerability discovery, wasting considerable effort on non-vulnerable code regions, and (ii) lack semantic understanding of stateful contracts, generating numerous invalid transaction sequences that cannot pass runtime execution.   In this paper, we propose SmartFuzz, a novel collaborative reflective fuzzer for smart contract vulnerability detection. It employs large language model-driven agents as the fuzzing engine and continuously improves itself by learning and reflecting through interactions with the environment. Specifically, we first propose a new Continuous Reflection Process (CRP) for fuzzing smart contracts, which reforms the transaction sequence generation as a self-evolving process through continuous reflection on feedback from the runtime environment. Then, we present the Reactive Collaborative Chain (RCC) to orchestrate the fuzzing process into multiple sub-tasks based on the dependencies of transaction sequences. Furthermore, we design a multi-agent collaborative team, where each expert agent is guided by the RCC to jointly generate and refine transaction sequences from both global and local perspectives. We conduct extensive experiments to evaluate SmartFuzz's performance on real-world contracts and DApp projects. The results demonstrate that SmartFuzz outperforms existing state-of-the-art tools: (i) it detects 5.8\%-74.7\% more vulnerabilities within 30 minutes, and (ii) it reduces false negatives by up to 80\%.

摘要: Fuzing是一种广泛使用的检测智能合约漏洞的技术，它生成交易序列来探索智能合约的执行路径。然而，现有的模糊器在检测复杂的漏洞方面表现不佳，这些漏洞需要具有适当输入来触发的特定攻击事务序列，因为它们（i）将代码覆盖的优先级置于漏洞发现之上，在非脆弱代码区域上浪费了大量精力，并且（ii）缺乏对有状态合同的语义理解，生成大量无法通过运行时执行的无效事务序列。   本文中，我们提出了SmartFuzz，这是一种用于智能合约漏洞检测的新型协作反射模糊器。它采用大型语言模型驱动的代理作为模糊引擎，并通过与环境的交互来学习和反思来不断改进自己。具体来说，我们首先提出了一种新的连续反射流程（CPR）来模糊智能合同，该流程通过对运行时环境的反馈的持续反射来将交易序列生成改革为一个自我进化的过程。然后，我们提出了反应式协作链（RNC），以根据事务序列的依赖性将模糊过程编排为多个子任务。此外，我们还设计了一个多代理协作团队，每个专家代理都在RNC的指导下，从全球和本地的角度共同生成和完善交易序列。我们进行了广泛的实验来评估SmartFuzz在现实世界合同和DApp项目中的表现。结果表明，SmartFuzz的性能优于现有的最先进工具：（i）它在30分钟内检测到了5.8%-74.7%的漏洞，（ii）它将误报率减少了高达80%。



## **15. Rethinking Deep Alignment Through The Lens Of Incomplete Learning**

从不完全学习的角度重新思考深度对齐 cs.LG

AAAI'26

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12155v1) [paper-pdf](https://arxiv.org/pdf/2511.12155v1)

**Authors**: Thong Bach, Dung Nguyen, Thao Minh Le, Truyen Tran

**Abstract**: Large language models exhibit systematic vulnerabilities to adversarial attacks despite extensive safety alignment. We provide a mechanistic analysis revealing that position-dependent gradient weakening during autoregressive training creates signal decay, leading to incomplete safety learning where safety training fails to transform model preferences in later response regions fully. We introduce base-favored tokens -- vocabulary elements where base models assign higher probability than aligned models -- as computational indicators of incomplete safety learning and develop a targeted completion method that addresses undertrained regions through adaptive penalties and hybrid teacher distillation. Experimental evaluation across Llama and Qwen model families demonstrates dramatic improvements in adversarial robustness, with 48--98% reductions in attack success rates while preserving general capabilities. These results establish both a mechanistic understanding and practical solutions for fundamental limitations in safety alignment methodologies.

摘要: 尽管进行了广泛的安全调整，大型语言模型仍表现出对对抗攻击的系统性漏洞。我们提供了一种机制分析，揭示了自回归训练期间与位置相关的梯度减弱会导致信号衰减，从而导致不完整的安全学习，即安全训练未能完全改变后期响应区域中的模型偏好。我们引入了基础偏好的代币（基础模型分配的概率比对齐模型更高的词汇元素）作为不完全安全学习的计算指标，并开发了一种有针对性的完成方法，通过自适应惩罚和混合教师提炼来解决训练不足的区域。Llama和Qwen模型家族的实验评估表明，对抗稳健性有了显着提高，攻击成功率降低了48- 98%，同时保留了一般能力。这些结果为安全对齐方法的基本局限性建立了机械性的理解和实用的解决方案。



## **16. AttackVLA: Benchmarking Adversarial and Backdoor Attacks on Vision-Language-Action Models**

AttackVLA：视觉-语言-动作模型上的对抗性和后门攻击基准 cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12149v1) [paper-pdf](https://arxiv.org/pdf/2511.12149v1)

**Authors**: Jiayu Li, Yunhan Zhao, Xiang Zheng, Zonghuan Xu, Yige Li, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Vision-Language-Action (VLA) models enable robots to interpret natural-language instructions and perform diverse tasks, yet their integration of perception, language, and control introduces new safety vulnerabilities. Despite growing interest in attacking such models, the effectiveness of existing techniques remains unclear due to the absence of a unified evaluation framework. One major issue is that differences in action tokenizers across VLA architectures hinder reproducibility and fair comparison. More importantly, most existing attacks have not been validated in real-world scenarios. To address these challenges, we propose AttackVLA, a unified framework that aligns with the VLA development lifecycle, covering data construction, model training, and inference. Within this framework, we implement a broad suite of attacks, including all existing attacks targeting VLAs and multiple adapted attacks originally developed for vision-language models, and evaluate them in both simulation and real-world settings. Our analysis of existing attacks reveals a critical gap: current methods tend to induce untargeted failures or static action states, leaving targeted attacks that drive VLAs to perform precise long-horizon action sequences largely unexplored. To fill this gap, we introduce BackdoorVLA, a targeted backdoor attack that compels a VLA to execute an attacker-specified long-horizon action sequence whenever a trigger is present. We evaluate BackdoorVLA in both simulated benchmarks and real-world robotic settings, achieving an average targeted success rate of 58.4% and reaching 100% on selected tasks. Our work provides a standardized framework for evaluating VLA vulnerabilities and demonstrates the potential for precise adversarial manipulation, motivating further research on securing VLA-based embodied systems.

摘要: 视觉-语言-动作（VLA）模型使机器人能够解释自然语言指令并执行各种任务，但它们对感知、语言和控制的集成引入了新的安全漏洞。尽管人们对攻击此类模型的兴趣越来越大，但由于缺乏统一的评估框架，现有技术的有效性仍然不清楚。一个主要问题是VLA架构中动作标记器的差异阻碍了可重复性和公平比较。更重要的是，大多数现有的攻击尚未在现实世界场景中得到验证。为了应对这些挑战，我们提出了AttackVLA，这是一个与VLA开发生命周期保持一致的统一框架，涵盖数据构建、模型训练和推理。在此框架内，我们实施了一系列广泛的攻击，包括所有针对VLA的现有攻击以及最初为视觉语言模型开发的多种改编攻击，并在模拟和现实环境中对其进行评估。我们对现有攻击的分析揭示了一个关键差距：当前的方法往往会引发无针对性的失败或静态动作状态，从而导致驱动VLA执行精确的长期动作序列的有针对性的攻击在很大程度上未被探索。为了填补这一空白，我们引入了BackdoorVLA，这是一种有针对性的后门攻击，它迫使VLA在出现触发器时执行攻击者指定的长期行动序列。我们在模拟基准测试和现实世界机器人环境中评估BackdoorVLA，实现了58.4%的平均目标成功率，并在选定任务中达到100%。我们的工作提供了一个用于评估VLA漏洞的标准化框架，并展示了精确对抗操纵的潜力，推动了对保护基于VLA的嵌入式系统的进一步研究。



## **17. BudgetLeak: Membership Inference Attacks on RAG Systems via the Generation Budget Side Channel**

BudgetLeak：通过发电预算侧渠道对RAG系统进行会员推断攻击 cs.CR

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.12043v1) [paper-pdf](https://arxiv.org/pdf/2511.12043v1)

**Authors**: Hao Li, Jiajun He, Guangshuo Wang, Dengguo Feng, Zheng Li, Min Zhang

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models by integrating external knowledge, but reliance on proprietary or sensitive corpora poses various data risks, including privacy leakage and unauthorized data usage. Membership inference attacks (MIAs) are a common technique to assess such risks, yet existing approaches underperform in RAG due to black-box constraints and the absence of strong membership signals. In this paper, we identify a previously unexplored side channel in RAG systems: the generation budget, which controls the maximum number of tokens allowed in a generated response. Varying this budget reveals observable behavioral patterns between member and non-member queries, as members gain quality more rapidly with larger budgets. Building on this insight, we propose BudgetLeak, a novel membership inference attack that probes responses under different budgets and analyzes metric evolution via sequence modeling or clustering. Extensive experiments across four datasets, three LLM generators, and two retrievers demonstrate that BudgetLeak consistently outperforms existing baselines, while maintaining high efficiency and practical viability. Our findings reveal a previously overlooked data risk in RAG systems and highlight the need for new defenses.

摘要: 检索增强生成（RAG）通过集成外部知识来增强大型语言模型，但依赖专有或敏感语料库会带来各种数据风险，包括隐私泄露和未经授权的数据使用。成员推理攻击（MIA）是一种常见的技术来评估这样的风险，但现有的方法在RAG由于黑盒约束和缺乏强大的成员信号表现不佳。在本文中，我们确定了一个以前未开发的侧通道RAG系统：生成预算，它控制的令牌允许在生成的响应的最大数量。改变这个预算揭示了成员和非成员查询之间可观察到的行为模式，因为成员在更大的预算下更快地获得质量。基于这一见解，我们提出了BudgetLeak，这是一种新型的成员资格推断攻击，可以探测不同预算下的响应，并通过序列建模或集群分析指标演变。针对四个数据集、三个LLM生成器和两个检索器的广泛实验表明，BudgetLeak始终优于现有基线，同时保持高效率和实际可行性。我们的研究结果揭示了RAG系统中以前被忽视的数据风险，并强调了对新防御措施的需求。



## **18. "Power of Words": Stealthy and Adaptive Private Information Elicitation via LLM Communication Strategies**

“言语的力量”：通过LLM传播策略秘密且自适应的私人信息获取 cs.HC

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2511.11961v1) [paper-pdf](https://arxiv.org/pdf/2511.11961v1)

**Authors**: Shuning Zhang, Jiaqi Bai, Linzhi Wang, Shixuan Li, Xin Yi, Hewu Li

**Abstract**: While communication strategies of Large Language Models (LLMs) are crucial for human-LLM interactions, they can also be weaponized to elicit private information, yet such stealthy attacks remain under-explored. This paper introduces the first adaptive attack framework for stealthy and targeted private information elicitation via communication strategies. Our framework operates in a dynamic closed-loop: it first performs real-time psychological profiling of the users' state, then adaptively selects an optimized communication strategy, and finally maintains stealthiness through prompt-based rewriting. We validated this framework through a user study (N=84), demonstrating its generalizability across 3 distinct LLMs and 3 scenarios. The targeted attacks achieved a 205.4% increase in eliciting specific targeted information compared to stealthy interactions without strategies. Even stealthy interactions without specific strategies successfully elicited private information in 54.8% cases. Notably, users not only failed to detect the manipulation but paradoxically rated the attacking chatbot as more empathetic and trustworthy. Finally, we advocate for mitigations, encouraging developers to integrate adaptive, just-in-time alerts, users to build literacy against specific manipulative tactics, and regulators to define clear ethical boundaries distinguishing benign persuasion from coercion.

摘要: 虽然大型语言模型（LLM）的通信策略对于人与LLM的交互至关重要，但它们也可以被武器化以获取私人信息，但此类隐形攻击仍然没有得到充分的探索。本文介绍了第一个自适应攻击框架，用于通过通信策略秘密和有针对性地获取私人信息。我们的框架以动态闭环方式运行：它首先对用户状态进行实时心理分析，然后自适应地选择优化的通信策略，最后通过基于预算的重写保持隐身性。我们通过用户研究（N=84）验证了该框架，展示了其在3种不同的LLM和3种场景中的通用性。与没有策略的隐形互动相比，有针对性的攻击在获取特定目标信息方面增加了205.4%。即使是没有具体策略的秘密互动，也有54.8%的案例成功获取了私人信息。值得注意的是，用户不仅没有检测到操纵，而且自相矛盾地认为攻击聊天机器人更有同理心和值得信赖。最后，我们主张采取缓解措施，鼓励开发人员集成自适应的及时警报，鼓励用户提高针对特定操纵策略的素养，并鼓励监管机构定义明确的道德界限，区分善意说服与胁迫。



## **19. SEAL: Subspace-Anchored Watermarks for LLM Ownership**

SEAL：LLM所有权的子空间锚定水印 cs.CR

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11356v1) [paper-pdf](https://arxiv.org/pdf/2511.11356v1)

**Authors**: Yanbo Dai, Zongjie Li, Zhenlan Ji, Shuai Wang

**Abstract**: Large language models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks, demonstrating human-level performance in text generation, reasoning, and question answering. However, training such models requires substantial computational resources, large curated datasets, and sophisticated alignment procedures. As a result, they constitute highly valuable intellectual property (IP) assets that warrant robust protection mechanisms. Existing IP protection approaches suffer from critical limitations. Model fingerprinting techniques can identify model architectures but fail to establish ownership of specific model instances. In contrast, traditional backdoor-based watermarking methods embed behavioral anomalies that can be easily removed through common post-processing operations such as fine-tuning or knowledge distillation.   We propose SEAL, a subspace-anchored watermarking framework that embeds multi-bit signatures directly into the model's latent representational space, supporting both white-box and black-box verification scenarios. Our approach leverages model editing techniques to align the hidden representations of selected anchor samples with predefined orthogonal bit vectors. This alignment embeds the watermark while preserving the model's original factual predictions, rendering the watermark functionally harmless and stealthy. We conduct comprehensive experiments on multiple benchmark datasets and six prominent LLMs, comparing SEAL with 11 existing fingerprinting and watermarking methods to demonstrate its superior effectiveness, fidelity, efficiency, and robustness. Furthermore, we evaluate SEAL under potential knowledgeable attacks and show that it maintains strong verification performance even when adversaries possess knowledge of the watermarking mechanism and the embedded signatures.

摘要: 大型语言模型（LLM）在广泛的自然语言处理任务中取得了显着的成功，展示了文本生成、推理和问答方面的人类水平性能。然而，训练此类模型需要大量的计算资源、大型精心策划的数据集和复杂的对齐程序。因此，它们构成了非常有价值的知识产权（IP）资产，需要强有力的保护机制。现有的知识产权保护方法存在严重局限性。模型指纹技术可以识别模型架构，但无法建立特定模型实例的所有权。相比之下，传统的基于后门的水印方法嵌入了行为异常，可以通过微调或知识蒸馏等常见的后处理操作轻松删除这些异常。   我们提出SEAL，这是一种子空间锚定的水印框架，它将多位签名直接嵌入到模型的潜在代表空间中，支持白盒和黑盒验证场景。我们的方法利用模型编辑技术，将选定的锚样本的隐藏表示与预定义的正交位向量对齐。这种对齐嵌入水印，同时保留模型的原始事实预测，使水印功能无害和隐形。我们在多个基准数据集和六个突出的LLM上进行了全面的实验，将SEAL与11种现有的指纹和水印方法进行了比较，以证明其优越的有效性，保真度，效率和鲁棒性。此外，我们评估SEAL下潜在的知识型攻击，并表明它保持强大的验证性能，即使当对手拥有知识的水印机制和嵌入式签名。



## **20. Analysing Personal Attacks in U.S. Presidential Debates**

分析美国总统辩论中的人身攻击 cs.CL

13 pages

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11108v1) [paper-pdf](https://arxiv.org/pdf/2511.11108v1)

**Authors**: Ruban Goyal, Rohitash Chandra, Sonit Singh

**Abstract**: Personal attacks have become a notable feature of U.S. presidential debates and play an important role in shaping public perception during elections. Detecting such attacks can improve transparency in political discourse and provide insights for journalists, analysts and the public. Advances in deep learning and transformer-based models, particularly BERT and large language models (LLMs) have created new opportunities for automated detection of harmful language. Motivated by these developments, we present a framework for analysing personal attacks in U.S. presidential debates. Our work involves manual annotation of debate transcripts across the 2016, 2020 and 2024 election cycles, followed by statistical and language-model based analysis. We investigate the potential of fine-tuned transformer models alongside general-purpose LLMs to detect personal attacks in formal political speech. This study demonstrates how task-specific adaptation of modern language models can contribute to a deeper understanding of political communication.

摘要: 人身攻击已成为美国总统辩论的一个显着特征，并在选举期间塑造公众看法方面发挥着重要作用。检测此类攻击可以提高政治话语的透明度，并为记者、分析师和公众提供见解。深度学习和基于转换器的模型，特别是BERT和大型语言模型（LLM）的进步，为自动检测有害语言创造了新的机会。受这些事态发展的启发，我们提出了一个分析美国总统辩论中人身攻击的框架。我们的工作涉及对2016年、2020年和2024年选举周期的辩论记录进行手动注释，然后进行基于统计和语言模型的分析。我们研究了微调Transformer模型与通用LLM一起检测正式政治演讲中的人身攻击的潜力。这项研究展示了现代语言模型的针对特定任务的调整如何有助于更深入地理解政治沟通。



## **21. Data Poisoning Vulnerabilities Across Healthcare AI Architectures: A Security Threat Analysis**

医疗保健人工智能架构中的数据中毒漏洞：安全威胁分析 cs.CR

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.11020v1) [paper-pdf](https://arxiv.org/pdf/2511.11020v1)

**Authors**: Farhad Abtahi, Fernando Seoane, Iván Pau, Mario Vega-Barbas

**Abstract**: Healthcare AI systems face major vulnerabilities to data poisoning that current defenses and regulations cannot adequately address. We analyzed eight attack scenarios in four categories: architectural attacks on convolutional neural networks, large language models, and reinforcement learning agents; infrastructure attacks exploiting federated learning and medical documentation systems; critical resource allocation attacks affecting organ transplantation and crisis triage; and supply chain attacks targeting commercial foundation models. Our findings indicate that attackers with access to only 100-500 samples can compromise healthcare AI regardless of dataset size, often achieving over 60 percent success, with detection taking an estimated 6 to 12 months or sometimes not occurring at all. The distributed nature of healthcare infrastructure creates many entry points where insiders with routine access can launch attacks with limited technical skill. Privacy laws such as HIPAA and GDPR can unintentionally shield attackers by restricting the analyses needed for detection. Supply chain weaknesses allow a single compromised vendor to poison models across 50 to 200 institutions. The Medical Scribe Sybil scenario shows how coordinated fake patient visits can poison data through legitimate clinical workflows without requiring a system breach. Current regulations lack mandatory adversarial robustness testing, and federated learning can worsen risks by obscuring attribution. We recommend multilayer defenses including required adversarial testing, ensemble-based detection, privacy-preserving security mechanisms, and international coordination on AI security standards. We also question whether opaque black-box models are suitable for high-stakes clinical decisions, suggesting a shift toward interpretable systems with verifiable safety guarantees.

摘要: 医疗保健人工智能系统面临着数据中毒的重大漏洞，当前的防御和法规无法充分解决这些漏洞。我们分析了四类八种攻击场景：对卷积神经网络、大型语言模型和强化学习代理的架构攻击;利用联邦学习和医疗文档系统的基础设施攻击;影响器官移植和危机分诊的关键资源分配攻击;以及针对商业基金会模型的供应链攻击。我们的研究结果表明，无论数据集大小如何，仅访问100-500个样本的攻击者都可以损害医疗保健人工智能，通常成功率超过60%，检测估计需要6至12个月，有时根本不发生。医疗保健基础设施的分布式性质创造了许多切入点，拥有常规访问权限的内部人员可以以有限的技术技能发起攻击。HIPAA和GDPR等隐私法可以通过限制检测所需的分析来无意中保护攻击者。供应链的弱点允许一个受影响的供应商毒害50至200个机构的模型。Medical Scribe Sybil场景展示了协调的虚假患者就诊如何通过合法的临床工作流程毒害数据，而不需要系统漏洞。当前的法规缺乏强制的对抗稳健性测试，联邦学习可能会通过模糊归因而加剧风险。我们建议多层防御，包括所需的对抗测试、基于集成的检测、隐私保护的安全机制以及人工智能安全标准的国际协调。我们还质疑不透明的黑匣子模型是否适合高风险的临床决策，建议转向具有可验证安全保证的可解释系统。



## **22. Synthetic Voices, Real Threats: Evaluating Large Text-to-Speech Models in Generating Harmful Audio**

合成语音，真实威胁：评估生成有害音频的大型文本到语音模型 cs.SD

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.10913v1) [paper-pdf](https://arxiv.org/pdf/2511.10913v1)

**Authors**: Guangke Chen, Yuhui Wang, Shouling Ji, Xiapu Luo, Ting Wang

**Abstract**: Modern text-to-speech (TTS) systems, particularly those built on Large Audio-Language Models (LALMs), generate high-fidelity speech that faithfully reproduces input text and mimics specified speaker identities. While prior misuse studies have focused on speaker impersonation, this work explores a distinct content-centric threat: exploiting TTS systems to produce speech containing harmful content. Realizing such threats poses two core challenges: (1) LALM safety alignment frequently rejects harmful prompts, yet existing jailbreak attacks are ill-suited for TTS because these systems are designed to faithfully vocalize any input text, and (2) real-world deployment pipelines often employ input/output filters that block harmful text and audio.   We present HARMGEN, a suite of five attacks organized into two families that address these challenges. The first family employs semantic obfuscation techniques (Concat, Shuffle) that conceal harmful content within text. The second leverages audio-modality exploits (Read, Spell, Phoneme) that inject harmful content through auxiliary audio channels while maintaining benign textual prompts. Through evaluation across five commercial LALMs-based TTS systems and three datasets spanning two languages, we demonstrate that our attacks substantially reduce refusal rates and increase the toxicity of generated speech.   We further assess both reactive countermeasures deployed by audio-streaming platforms and proactive defenses implemented by TTS providers. Our analysis reveals critical vulnerabilities: deepfake detectors underperform on high-fidelity audio; reactive moderation can be circumvented by adversarial perturbations; while proactive moderation detects 57-93% of attacks. Our work highlights a previously underexplored content-centric misuse vector for TTS and underscore the need for robust cross-modal safeguards throughout training and deployment.

摘要: 现代文本转语音（TTC）系统，特别是那些建立在大型音频语言模型（LALM）上的系统，会生成高保真语音，忠实地再现输入文本并模仿指定的说话者身份。虽然之前的滥用研究重点是说话者模仿，但这项工作探索了一种独特的以内容为中心的威胁：利用TTC系统产生包含有害内容的语音。实现此类威胁带来了两个核心挑战：（1）LALM安全对齐经常拒绝有害提示，但现有的越狱攻击不适合TTC，因为这些系统旨在忠实地发声任何输入文本，以及（2）现实世界的部署管道通常使用阻止有害文本和音频的输入/输出过滤器。   我们介绍了HARMGEN，这是一系列五次袭击，分为两个系列，旨在应对这些挑战。第一个家族采用语义混淆技术（Concat、Shuffle）来隐藏文本中的有害内容。第二种利用音频模式漏洞（Read、Spell、Phoneme），通过辅助音频通道注入有害内容，同时保持良性文本提示。通过对五个基于LALM的商业TTC系统和跨越两种语言的三个数据集的评估，我们证明我们的攻击大大降低了拒绝率并增加了生成语音的毒性。   我们进一步评估了音频流媒体平台部署的反应性对策和TTC提供商实施的主动防御。我们的分析揭示了关键漏洞：Deepfake检测器在高保真音频上表现不佳;对抗性干扰可以规避反应性审核;而主动审核可以检测57-93%的攻击。我们的工作强调了以前未充分探索的以内容为中心的TTC滥用载体，并强调了在整个培训和部署过程中对强有力的跨模式保护措施的必要性。



## **23. Say It Differently: Linguistic Styles as Jailbreak Vectors**

不同地说：作为越狱载体的语言风格 cs.CL

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.10519v1) [paper-pdf](https://arxiv.org/pdf/2511.10519v1)

**Authors**: Srikant Panda, Avinash Rai

**Abstract**: Large Language Models (LLMs) are commonly evaluated for robustness against paraphrased or semantically equivalent jailbreak prompts, yet little attention has been paid to linguistic variation as an attack surface. In this work, we systematically study how linguistic styles such as fear or curiosity can reframe harmful intent and elicit unsafe responses from aligned models. We construct style-augmented jailbreak benchmark by transforming prompts from 3 standard datasets into 11 distinct linguistic styles using handcrafted templates and LLM-based rewrites, while preserving semantic intent. Evaluating 16 open- and close-source instruction-tuned models, we find that stylistic reframing increases jailbreak success rates by up to +57 percentage points. Styles such as fearful, curious and compassionate are most effective and contextualized rewrites outperform templated variants.   To mitigate this, we introduce a style neutralization preprocessing step using a secondary LLM to strip manipulative stylistic cues from user inputs, significantly reducing jailbreak success rates. Our findings reveal a systemic and scaling-resistant vulnerability overlooked in current safety pipelines.

摘要: 大型语言模型（LLM）通常会针对重述或语义等效的越狱提示进行鲁棒性评估，但很少有人关注语言变化作为攻击面。在这项工作中，我们系统地研究恐惧或好奇心等语言风格如何重新定义有害意图并引发一致模型的不安全反应。我们通过使用手工制作的模板和基于LLM的重写将3个标准数据集的提示转换为11种不同的语言风格，同时保留语义意图，来构建风格增强的越狱基准。在评估16个开放和封闭源的描述调整模型后，我们发现风格重组可将越狱成功率提高高达+57个百分点。恐惧、好奇和富有同情心等风格是最有效的，并且背景化的重写优于模板化的变体。   为了缓解这一问题，我们引入了风格中和预处理步骤，使用二级LLM来从用户输入中去除操纵性风格线索，从而显着降低越狱成功率。我们的研究结果揭示了当前安全管道中忽视的系统性和抗扩展性漏洞。



## **24. BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models**

BadThink：引发对大型语言模型中思想链推理的过度思考攻击 cs.CR

Accepted at AAAI 2026 (Main Track). This arXiv version corresponds to the camera-ready manuscript and includes expanded appendices. Please cite the AAAI 2026 version when available

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.10714v1) [paper-pdf](https://arxiv.org/pdf/2511.10714v1)

**Authors**: Shuaitong Liu, Renjue Li, Lijia Yu, Lijun Zhang, Zhiming Liu, Gaojie Jin

**Abstract**: Recent advances in Chain-of-Thought (CoT) prompting have substantially improved the reasoning capabilities of large language models (LLMs), but have also introduced their computational efficiency as a new attack surface. In this paper, we propose BadThink, the first backdoor attack designed to deliberately induce "overthinking" behavior in CoT-enabled LLMs while ensuring stealth. When activated by carefully crafted trigger prompts, BadThink manipulates the model to generate inflated reasoning traces - producing unnecessarily redundant thought processes while preserving the consistency of final outputs. This subtle attack vector creates a covert form of performance degradation that significantly increases computational costs and inference time while remaining difficult to detect through conventional output evaluation methods. We implement this attack through a sophisticated poisoning-based fine-tuning strategy, employing a novel LLM-based iterative optimization process to embed the behavior by generating highly naturalistic poisoned data. Our experiments on multiple state-of-the-art models and reasoning tasks show that BadThink consistently increases reasoning trace lengths - achieving an over 17x increase on the MATH-500 dataset - while remaining stealthy and robust. This work reveals a critical, previously unexplored vulnerability where reasoning efficiency can be covertly manipulated, demonstrating a new class of sophisticated attacks against CoT-enabled systems.

摘要: 思想链（CoT）提示的最新进展大大提高了大型语言模型（LLM）的推理能力，但也引入了计算效率作为新的攻击面。在本文中，我们提出了BadThink，这是第一个后门攻击，旨在故意在支持CoT的LLM中诱导“过度思考”行为，同时确保隐身。当被精心设计的触发提示激活时，BadThink会操纵模型以生成膨胀的推理痕迹--在保持最终输出的一致性的同时产生不必要的冗余思维过程。这种微妙的攻击载体造成了一种隐蔽形式的性能退化，显着增加了计算成本和推理时间，同时仍然难以通过传统的输出评估方法检测到。我们通过复杂的基于中毒的微调策略来实现这种攻击，采用新型的基于LLM的迭代优化过程来通过生成高度自然主义的中毒数据来嵌入行为。我们对多个最先进模型和推理任务的实验表明，BadThink持续增加推理轨迹长度-在MAT-500数据集上实现了17倍以上的增长-同时保持隐蔽性和稳健性。这项工作揭示了一个以前未探索的关键漏洞，其中推理效率可以被秘密操纵，展示了针对支持CoT的系统的新型复杂攻击。



## **25. Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard**

对多模式LLM的语音音频合成攻击及其使用SALMONN-Guard的缓解 cs.SD

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2511.10222v2) [paper-pdf](https://arxiv.org/pdf/2511.10222v2)

**Authors**: Yudong Yang, Xuezhen Zhang, Zhifeng Han, Siyin Wang, Jimin Zhuang, Zengrui Jin, Jing Shao, Guangzhi Sun, Chao Zhang

**Abstract**: Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful.

摘要: 大型语言模型（LLM）的最新进展使人们能够理解语音和非语音音频，但也暴露了当前保护措施未充分处理的复杂音频输入所出现的新安全风险。我们引入SACRED-Bench（用于RED团队的语音音频合成）来评估LLM在复杂的基于音频的攻击下的稳健性。与现有的依赖于噪音优化或白盒访问的基于扰动的方法不同，SACRED-Bench利用了语音音频合成机制。SACRED-Bench采用三种机制：（a）语音重叠和多说话者对话，将有害提示嵌入良性语音之下或旁边;（b）语音音频混合，通过非语音音频与良性语音或音频一起暗示不安全意图;（c）规避纯文本过滤器的多种口语指令格式（开放式QA，是/否）。实验表明，即使是最先进的专有LLM Gemini 2.5 Pro，在SACRED-Bench测试集中仍然表现出66%的攻击成功率，暴露了跨模式、语音音频合成攻击下的漏洞。为了弥合这一差距，我们提出SALMONN-Guard，这是一种保护LLM，可联合检查语音、音频和文本以进行安全判断，将攻击成功率降低至20%。我们的结果凸显了为多模式LLM的安全性而需要音频感知防御。基准检查站和SALMONS-Guard检查站可在https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench上找到。警告：本文包含可能令人反感或有害的示例。



## **26. MTAttack: Multi-Target Backdoor Attacks against Large Vision-Language Models**

MTA ttack：针对大型视觉语言模型的多目标后门攻击 cs.CV

AAAI2026, with supplementary material

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.10098v1) [paper-pdf](https://arxiv.org/pdf/2511.10098v1)

**Authors**: Zihan Wang, Guansong Pang, Wenjun Miao, Jin Zheng, Xiao Bai

**Abstract**: Recent advances in Large Visual Language Models (LVLMs) have demonstrated impressive performance across various vision-language tasks by leveraging large-scale image-text pretraining and instruction tuning. However, the security vulnerabilities of LVLMs have become increasingly concerning, particularly their susceptibility to backdoor attacks. Existing backdoor attacks focus on single-target attacks, i.e., targeting a single malicious output associated with a specific trigger. In this work, we uncover multi-target backdoor attacks, where multiple independent triggers corresponding to different attack targets are added in a single pass of training, posing a greater threat to LVLMs in real-world applications. Executing such attacks in LVLMs is challenging since there can be many incorrect trigger-target mappings due to severe feature interference among different triggers. To address this challenge, we propose MTAttack, the first multi-target backdoor attack framework for enforcing accurate multiple trigger-target mappings in LVLMs. The core of MTAttack is a novel optimization method with two constraints, namely Proxy Space Partitioning constraint and Trigger Prototype Anchoring constraint. It jointly optimizes multiple triggers in the latent space, with each trigger independently mapping clean images to a unique proxy class while at the same time guaranteeing their separability. Experiments on popular benchmarks demonstrate a high success rate of MTAttack for multi-target attacks, substantially outperforming existing attack methods. Furthermore, our attack exhibits strong generalizability across datasets and robustness against backdoor defense strategies. These findings highlight the vulnerability of LVLMs to multi-target backdoor attacks and underscore the urgent need for mitigating such threats. Code is available at https://github.com/mala-lab/MTAttack.

摘要: 大型视觉语言模型（LVLM）的最新进展通过利用大规模图像-文本预训练和指令调优，在各种视觉语言任务中展示了令人印象深刻的性能。然而，LVLM的安全漏洞变得越来越令人担忧，特别是它们容易受到后门攻击。现有的后门攻击集中在单目标攻击上，即针对与特定触发器关联的单个恶意输出。在这项工作中，我们发现了多目标后门攻击，即在一次训练中添加对应不同攻击目标的多个独立触发器，对现实世界应用中的LVLM构成了更大的威胁。在LVLM中执行此类攻击具有挑战性，因为由于不同触发器之间的严重特征干扰，可能会出现许多不正确的攻击者目标映射。为了应对这一挑战，我们提出了MTA tack，这是第一个多目标后门攻击框架，用于在LVLM中实施准确的多个攻击者-目标映射。MTA ttack的核心是一种具有两个约束的新型优化方法，即代理空间分区约束和触发器原型锚定约束。它联合优化潜在空间中的多个触发器，每个触发器独立地将干净的图像映射到唯一的代理类，同时保证它们的可分离性。流行基准测试的实验表明，MTA ttack对多目标攻击的成功率很高，大大优于现有的攻击方法。此外，我们的攻击在数据集中表现出很强的通用性以及针对后门防御策略的鲁棒性。这些发现凸显了LVLM对多目标后门攻击的脆弱性，并强调了缓解此类威胁的迫切需要。代码可在https://github.com/mala-lab/MTAttack上获取。



## **27. Phantom Menace: Exploring and Enhancing the Robustness of VLA Models against Physical Sensor Attacks**

Phantom Menace：探索和增强VLA模型对物理传感器攻击的鲁棒性 cs.RO

Accepted by AAAI 2026

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.10008v1) [paper-pdf](https://arxiv.org/pdf/2511.10008v1)

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored.   To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel ``Real-Sim-Real'' framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.

摘要: 视觉-语言-动作（VLA）模型通过实现端到端的感知到动作管道，彻底改变了机器人系统，该管道集成了多种感官模式，例如由摄像机处理的视觉信号和由麦克风捕获的听觉信号。这种多模式集成使VLA模型能够使用不同的传感器数据流来解释复杂的现实世界环境。鉴于基于VLA的系统严重依赖感官输入，VLA模型对抗物理世界传感器攻击的安全性仍然严重不足。   为了弥补这一差距，我们首次对针对VLA的物理传感器攻击进行了系统研究，量化了传感器攻击的影响并调查VLA模型的防御。我们引入了一种新型的“Real-Sim-Real”框架，该框架自动模拟基于物理的传感器攻击载体，包括六次针对摄像头和两个针对麦克风的攻击，并在真实的机器人系统上对其进行验证。通过在不同攻击参数下对各种VLA架构和任务进行大规模评估，我们展示了显着的漏洞，其易感性模式揭示了对任务类型和模型设计的关键依赖性。我们进一步开发了一种基于对抗训练的防御，可以增强VLA对传感器攻击引起的分布外物理扰动的鲁棒性，同时保持模型性能。我们的研究结果揭示了迫切需要标准化的稳健性基准和缓解策略，以确保VLA在安全关键环境中的部署。



## **28. EnchTable: Unified Safety Alignment Transfer in Fine-tuned Large Language Models**

EnchTable：微调大型语言模型中的统一安全对齐转移 cs.CL

Accepted by IEEE Symposium on Security and Privacy (S&P) 2026

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09880v1) [paper-pdf](https://arxiv.org/pdf/2511.09880v1)

**Authors**: Jialin Wu, Kecen Li, Zhicong Huang, Xinfeng Li, Xiaofeng Wang, Cheng Hong

**Abstract**: Many machine learning models are fine-tuned from large language models (LLMs) to achieve high performance in specialized domains like code generation, biomedical analysis, and mathematical problem solving. However, this fine-tuning process often introduces a critical vulnerability: the systematic degradation of safety alignment, undermining ethical guidelines and increasing the risk of harmful outputs. Addressing this challenge, we introduce EnchTable, a novel framework designed to transfer and maintain safety alignment in downstream LLMs without requiring extensive retraining. EnchTable leverages a Neural Tangent Kernel (NTK)-based safety vector distillation method to decouple safety constraints from task-specific reasoning, ensuring compatibility across diverse model architectures and sizes. Additionally, our interference-aware merging technique effectively balances safety and utility, minimizing performance compromises across various task domains. We implemented a fully functional prototype of EnchTable on three different task domains and three distinct LLM architectures, and evaluated its performance through extensive experiments on eleven diverse datasets, assessing both utility and model safety. Our evaluations include LLMs from different vendors, demonstrating EnchTable's generalization capability. Furthermore, EnchTable exhibits robust resistance to static and dynamic jailbreaking attacks, outperforming vendor-released safety models in mitigating adversarial prompts. Comparative analyses with six parameter modification methods and two inference-time alignment baselines reveal that EnchTable achieves a significantly lower unsafe rate, higher utility score, and universal applicability across different task domains. Additionally, we validate EnchTable can be seamlessly integrated into various deployment pipelines without significant overhead.

摘要: 许多机器学习模型都是从大型语言模型（LLM）进行微调的，以在代码生成、生物医学分析和数学问题解决等专业领域实现高性能。然而，这种微调过程往往会引入一个关键的漏洞：安全一致性的系统性退化，破坏道德准则并增加有害输出的风险。为了应对这一挑战，我们引入了EnchTable，这是一个新颖的框架，旨在转移和维护下游LLM的安全一致，而无需进行广泛的再培训。EnchTable利用基于神经切向核（NTK）的安全向量提炼方法将安全约束与特定任务推理脱钩，确保不同模型架构和大小之间的兼容性。此外，我们的干扰感知合并技术有效地平衡了安全性和实用性，最大限度地减少了各个任务域的性能损害。我们在三个不同的任务域和三个不同的LLM架构上实现了功能齐全的EnchTable原型，并通过对十一个不同数据集的广泛实验评估了其性能，评估了效用和模型安全性。我们的评估包括来自不同供应商的LLM，展示了EnchTable的概括能力。此外，EnchTable对静态和动态越狱攻击表现出强大的抵抗力，在减轻对抗提示方面优于供应商发布的安全模型。对六种参数修改方法和两种推断时间对齐基线的比较分析表明，EnchTable实现了显着较低的不安全率、较高的效用评分以及跨不同任务领域的普遍适用性。此外，我们还验证了EnchTable可以无缝集成到各种部署管道中，而无需承担重大费用。



## **29. Unlearning Imperative: Securing Trustworthy and Responsible LLMs through Engineered Forgetting**

放弃学习势在必行：通过精心设计的遗忘确保值得信赖和负责任的LLM cs.LG

14 pages, 4 figures, 4 tables

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09855v1) [paper-pdf](https://arxiv.org/pdf/2511.09855v1)

**Authors**: James Jin Kang, Dang Bui, Thanh Pham, Huo-Chong Ling

**Abstract**: The growing use of large language models in sensitive domains has exposed a critical weakness: the inability to ensure that private information can be permanently forgotten. Yet these systems still lack reliable mechanisms to guarantee that sensitive information can be permanently removed once it has been used. Retraining from the beginning is prohibitively costly, and existing unlearning methods remain fragmented, difficult to verify, and often vulnerable to recovery. This paper surveys recent research on machine unlearning for LLMs and considers how far current approaches can address these challenges. We review methods for evaluating whether forgetting has occurred, the resilience of unlearned models against adversarial attacks, and mechanisms that can support user trust when model complexity or proprietary limits restrict transparency. Technical solutions such as differential privacy, homomorphic encryption, federated learning, and ephemeral memory are examined alongside institutional safeguards including auditing practices and regulatory frameworks. The review finds steady progress, but robust and verifiable unlearning is still unresolved. Efficient techniques that avoid costly retraining, stronger defenses against adversarial recovery, and governance structures that reinforce accountability are needed if LLMs are to be deployed safely in sensitive applications. By integrating technical and organizational perspectives, this study outlines a pathway toward AI systems that can be required to forget, while maintaining both privacy and public trust.

摘要: 大型语言模型在敏感领域的使用越来越多，暴露了一个关键弱点：无法确保私人信息可以被永久遗忘。然而，这些系统仍然缺乏可靠的机制来保证敏感信息一旦被使用就可以被永久删除。从一开始的再培训成本高得令人望而却步，而且现有的学习方法仍然支离破碎、难以验证，并且往往很容易恢复。本文调查了最近关于LLM机器去学习的研究，并考虑了当前方法可以在多大程度上解决这些挑战。我们回顾了评估是否发生遗忘的方法、未学习的模型对抗对抗攻击的弹性，以及在模型复杂性或专有限制限制透明度时可以支持用户信任的机制。与审计实践和监管框架等机构保障措施一起审查了差异隐私、同质加密、联邦学习和短暂记忆等技术解决方案。审查发现取得了稳步进展，但稳健且可验证的取消学习仍未得到解决。如果要在敏感应用程序中安全部署LLM，就需要避免昂贵的再培训的高效技术、针对对抗性恢复的更强防御以及加强问责制的治理结构。通过整合技术和组织角度，这项研究概述了一条通往人工智能系统的途径，这些系统可能被要求忘记，同时维护隐私和公众信任。



## **30. Hail to the Thief: Exploring Attacks and Defenses in Decentralised GRPO**

向小偷致敬：探索分散式GRPO中的攻击和防御 cs.LG

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.09780v1) [paper-pdf](https://arxiv.org/pdf/2511.09780v1)

**Authors**: Nikolay Blagoev, Oğuzhan Ersoy, Lydia Yiyu Chen

**Abstract**: Group Relative Policy Optimization (GRPO) has demonstrated great utilization in post-training of Large Language Models (LLMs). In GRPO, prompts are answered by the model and, through reinforcement learning, preferred completions are learnt. Owing to the small communication volume, GRPO is inherently suitable for decentralised training as the prompts can be concurrently answered by multiple nodes and then exchanged in the forms of strings. In this work, we present the first adversarial attack in decentralised GRPO. We demonstrate that malicious parties can poison such systems by injecting arbitrary malicious tokens in benign models in both out-of-context and in-context attacks. Using empirical examples of math and coding tasks, we show that adversarial attacks can easily poison the benign nodes, polluting their local LLM post-training, achieving attack success rates up to 100% in as few as 50 iterations. We propose two ways to defend against these attacks, depending on whether all users train the same model or different models. We show that these defenses can achieve stop rates of up to 100%, making the attack impossible.

摘要: 组相对策略优化（GRPO）在大型语言模型（LLM）的后训练中表现出了巨大的利用率。在GRPO中，模型回答提示，并通过强化学习首选的完成。由于通信量小，GRPO本质上适合去中心化训练，因为提示可以由多个节点同时回答，然后以字符串的形式交换。在这项工作中，我们展示了去中心化GRPO中的第一次对抗攻击。我们证明，恶意方可以在上下文外和上下文内攻击中通过在良性模型中注入任意恶意令牌来毒害此类系统。使用数学和编码任务的经验示例，我们表明对抗性攻击很容易毒害良性节点，污染其本地LLM后训练，在短短50次迭代中实现高达100%的攻击成功率。我们提出了两种防御这些攻击的方法，具体取决于所有用户是训练相同的模型还是不同的模型。我们表明，这些防御措施可以实现高达100%的停止率，使攻击变得不可能。



## **31. Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models**

对生成性基因组模型的生物知情混合成员推断攻击 cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.07503v2) [paper-pdf](https://arxiv.org/pdf/2511.07503v2)

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs.

摘要: 遗传数据可用性的增加改变了基因组学研究，但由于其敏感性，对其处理提出了许多隐私问题。这项工作探索了使用语言模型（LM）来生成合成基因突变谱，利用差异隐私（DP）来保护敏感遗传数据。我们通过引入一种新型的生物知情混合成员推断攻击（biHMIA）来经验性地评估DP模式的隐私保证，该攻击将传统的黑匣子MIA与上下文基因组学指标相结合，以增强攻击能力。我们的实验表明，小型和大型Transformer GPT类模型都是小规模基因组学的可行合成变体生成器，并且与传统的基于度量的MIA相比，我们的混合攻击平均会导致更高的对抗成功。



## **32. Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment**

差异化定向干预避免LLM安全一致的框架 cs.CR

AAAI-26-AIA

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.06852v3) [paper-pdf](https://arxiv.org/pdf/2511.06852v3)

**Authors**: Peng Zhang, Peijie Sun

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.

摘要: 安全一致为大型语言模型（LLM）灌输了拒绝恶意请求的关键能力。之前的作品将这种拒绝机制建模为激活空间中的单一线性方向。我们认为这是一种过于简单化的做法，将两个功能上不同的神经过程混为一谈：伤害的检测和拒绝的执行。在这项工作中，我们将这个单一的表示解构为伤害检测方向和拒绝执行方向。利用这个细粒度模型，我们引入了差异双向干预（DBDI），这是一种新的白盒框架，可以精确地中和关键层的安全对齐。DBDI对拒绝执行方向应用自适应投影无效，同时通过直接转向抑制伤害检测方向。大量实验表明，DBDI优于著名的越狱方法，对Llama-2等模型的攻击成功率高达97.88%。通过提供更细粒度和机械化的框架，我们的工作为深入了解LLM安全对齐提供了新的方向。



## **33. Backdoor Attacks Against Speech Language Models**

针对语音语言模型的后门攻击 cs.CL

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2510.01157v2) [paper-pdf](https://arxiv.org/pdf/2510.01157v2)

**Authors**: Alexandrine Fortier, Thomas Thebaud, Jesús Villalba, Najim Dehak, Patrick Cardinal

**Abstract**: Large Language Models (LLMs) and their multimodal extensions are becoming increasingly popular. One common approach to enable multimodality is to cascade domain-specific encoders with an LLM, making the resulting model inherit vulnerabilities from all of its components. In this work, we present the first systematic study of audio backdoor attacks against speech language models. We demonstrate its effectiveness across four speech encoders and three datasets, covering four tasks: automatic speech recognition (ASR), speech emotion recognition, and gender and age prediction. The attack consistently achieves high success rates, ranging from 90.76% to 99.41%. To better understand how backdoors propagate, we conduct a component-wise analysis to identify the most vulnerable stages of the pipeline. Finally, we propose a fine-tuning-based defense that mitigates the threat of poisoned pretrained encoders.

摘要: 大型语言模型（LLM）及其多模式扩展正变得越来越受欢迎。启用多模式的一种常见方法是将特定于域的编码器与LLM级联，使生成的模型继承其所有组件的漏洞。在这项工作中，我们首次对针对语音语言模型的音频后门攻击进行了系统研究。我们在四个语音编码器和三个数据集中展示了它的有效性，涵盖四项任务：自动语音识别（ASB）、语音情感识别以及性别和年龄预测。该攻击的成功率始终很高，范围从90.76%到99.41%。为了更好地了解后门如何传播，我们进行了组件级分析，以识别管道中最脆弱的阶段。最后，我们提出了一种基于微调的防御，可以减轻中毒的预训练编码器的威胁。



## **34. SecInfer: Preventing Prompt Injection via Inference-time Scaling**

SecInfer：通过推理时缩放防止提示注入 cs.CR

**SubmitDate**: 2025-11-14    [abs](http://arxiv.org/abs/2509.24967v4) [paper-pdf](https://arxiv.org/pdf/2509.24967v4)

**Authors**: Yupei Liu, Yanting Wang, Yuqi Jia, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Prompt injection attacks pose a pervasive threat to the security of Large Language Models (LLMs). State-of-the-art prevention-based defenses typically rely on fine-tuning an LLM to enhance its security, but they achieve limited effectiveness against strong attacks. In this work, we propose \emph{SecInfer}, a novel defense against prompt injection attacks built on \emph{inference-time scaling}, an emerging paradigm that boosts LLM capability by allocating more compute resources for reasoning during inference. SecInfer consists of two key steps: \emph{system-prompt-guided sampling}, which generates multiple responses for a given input by exploring diverse reasoning paths through a varied set of system prompts, and \emph{target-task-guided aggregation}, which selects the response most likely to accomplish the intended task. Extensive experiments show that, by leveraging additional compute at inference, SecInfer effectively mitigates both existing and adaptive prompt injection attacks, outperforming state-of-the-art defenses as well as existing inference-time scaling approaches.

摘要: 提示注入攻击对大型语言模型（LLM）的安全性构成普遍威胁。最先进的基于预防的防御通常依赖于对LLM进行微调来增强其安全性，但它们对强攻击的有效性有限。在这项工作中，我们提出了\{SecInfer}，这是一种基于\{推断时间缩放}的新型防御方法，这是一种新兴范式，通过在推断期间分配更多计算资源进行推理来增强LLM能力。SecInfer由两个关键步骤组成：\{系统提示引导采样}，通过通过不同的系统提示集探索不同的推理路径，为给定输入生成多个响应，以及\{target-task-guided aggregage}，选择最有可能完成预期任务的响应。大量实验表明，通过在推理时利用额外的计算，SecInfer有效地减轻了现有的和自适应的即时注入攻击，性能优于最先进的防御以及现有的推理时扩展方法。



## **35. Automated Vulnerability Validation and Verification: A Large Language Model Approach**

自动化漏洞验证和验证：大型语言模型方法 cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2509.24037v2) [paper-pdf](https://arxiv.org/pdf/2509.24037v2)

**Authors**: Alireza Lotfi, Charalampos Katsis, Elisa Bertino

**Abstract**: Software vulnerabilities remain a critical security challenge, providing entry points for attackers into enterprise networks. Despite advances in security practices, the lack of high-quality datasets capturing diverse exploit behavior limits effective vulnerability assessment and mitigation. This paper introduces an end-to-end multi-step pipeline leveraging generative AI, specifically large language models (LLMs), to address the challenges of orchestrating and reproducing attacks to known software vulnerabilities. Our approach extracts information from CVE disclosures in the National Vulnerability Database, augments it with external public knowledge (e.g., threat advisories, code snippets) using Retrieval-Augmented Generation (RAG), and automates the creation of containerized environments and exploit code for each vulnerability. The pipeline iteratively refines generated artifacts, validates attack success with test cases, and supports complex multi-container setups. Our methodology overcomes key obstacles, including noisy and incomplete vulnerability descriptions, by integrating LLMs and RAG to fill information gaps. We demonstrate the effectiveness of our pipeline across different vulnerability types, such as memory overflows, denial of service, and remote code execution, spanning diverse programming languages, libraries and years. In doing so, we uncover significant inconsistencies in CVE descriptions, emphasizing the need for more rigorous verification in the CVE disclosure process. Our approach is model-agnostic, working across multiple LLMs, and we open-source the artifacts to enable reproducibility and accelerate security research. To the best of our knowledge, this is the first system to systematically orchestrate and exploit known vulnerabilities in containerized environments by combining general-purpose LLM reasoning with CVE data and RAG-based context enrichment.

摘要: 软件漏洞仍然是一个关键的安全挑战，为攻击者进入企业网络提供了切入点。尽管安全实践取得了进步，但缺乏捕捉不同利用行为的高质量数据集限制了有效的漏洞评估和缓解。本文介绍了一种端到端多步骤管道，利用生成式人工智能，特别是大型语言模型（LLM），来解决策划和复制对已知软件漏洞的攻击的挑战。我们的方法从国家漏洞数据库中的UTE披露中提取信息，并利用外部公共知识对其进行增强（例如，威胁数据库、代码片段），并自动创建容器化环境和针对每个漏洞的攻击代码。管道迭代地细化生成的工件，使用测试用例验证攻击成功，并支持复杂的多容器设置。我们的方法克服了关键的障碍，包括嘈杂和不完整的漏洞描述，通过集成LLM和RAG，以填补信息空白。我们展示了我们的管道跨不同漏洞类型的有效性，例如内存溢出，拒绝服务和远程代码执行，跨越不同的编程语言，库和年份。在此过程中，我们发现了UTE描述中的显着不一致之处，强调了在UTE披露过程中进行更严格验证的必要性。我们的方法是模型不可知的，跨多个LLM工作，并且我们开源工件以实现可重复性并加速安全研究。据我们所知，这是第一个通过将通用LLM推理与UTE数据和基于RAG的上下文丰富相结合来系统性地编排和利用容器化环境中已知漏洞的系统。



## **36. Jailbreaking LLMs via Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge**

通过具有针对性有毒知识的语义相关嵌套场景越狱LLM cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2510.01223v2) [paper-pdf](https://arxiv.org/pdf/2510.01223v2)

**Authors**: Ning Xu, Bo Gao, Hui Dou

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks. However, they remain exposed to jailbreak attacks, eliciting harmful responses. The nested scenario strategy has been increasingly adopted across various methods, demonstrating immense potential. Nevertheless, these methods are easily detectable due to their prominent malicious intentions. In this work, we are the first to find and systematically verify that LLMs' alignment defenses are not sensitive to nested scenarios, where these scenarios are highly semantically relevant to the queries and incorporate targeted toxic knowledge. This is a crucial yet insufficiently explored direction. Based on this, we propose RTS-Attack (Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge), an adaptive and automated framework to examine LLMs' alignment. By building scenarios highly relevant to the queries and integrating targeted toxic knowledge, RTS-Attack bypasses the alignment defenses of LLMs. Moreover, the jailbreak prompts generated by RTS-Attack are free from harmful queries, leading to outstanding concealment. Extensive experiments demonstrate that RTS-Attack exhibits superior performance in both efficiency and universality compared to the baselines across diverse advanced LLMs, including GPT-4o, Llama3-70b, and Gemini-pro. Our complete code is available at https://github.com/nercode/Work. WARNING: THIS PAPER CONTAINS POTENTIALLY HARMFUL CONTENT.

摘要: 大型语言模型（LLM）在各种任务中表现出了非凡的能力。然而，他们仍然面临越狱攻击，引发有害反应。嵌套场景策略越来越多地被各种方法采用，展现出巨大的潜力。然而，这些方法由于其明显的恶意意图而很容易被检测到。在这项工作中，我们是第一个发现并系统地验证LLM的对齐防御对嵌套场景不敏感的人，这些场景与查询在语义上高度相关，并包含有针对性的有毒知识。这是一个至关重要但尚未充分探索的方向。基于此，我们提出了RTS-Attack（具有目标有毒知识的语义相关嵌套场景），这是一个自适应的自动化框架，用于检查LLM的一致性。通过构建与查询高度相关的场景并集成有针对性的有毒知识，RTS-Attack绕过了LLM的对齐防御。此外，RTS-Attack生成的越狱提示没有有害查询，具有出色的隐蔽性。大量实验表明，与GPT-4 o、Llama 3 - 70 b和Gemini-pro等各种高级LLM的基线相比，RTS-Attack在效率和通用性方面都表现出卓越的性能。我们的完整代码可在https://github.com/nercode/Work上获取。警告：本文包含潜在有害内容。



## **37. From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing**

从能力到性能：在渗透测试中评估LLM架构的关键功能属性 cs.AI

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2509.14289v3) [paper-pdf](https://arxiv.org/pdf/2509.14289v3)

**Authors**: Lanxiao Huang, Daksh Dave, Tyler Cody, Peter Beling, Ming Jin

**Abstract**: Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.

摘要: 大型语言模型（LLM）越来越多地用于自动化或增强渗透测试，但它们在攻击阶段的有效性和可靠性仍不清楚。我们在现实的渗透测试场景中对多个基于LLM的代理（从单代理到模块化设计）进行了全面评估，测量经验性能和反复出现的故障模式。我们还通过有针对性的增强来隔离五种核心功能能力的影响：全球上下文记忆（GCM）、代理间消息传递（ILM）、上下文条件调用（CI）、自适应规划（AP）和实时监控（RTI）。这些干预措施分别支持：（i）上下文一致性和保留，（ii）组件间协调和状态管理，（iii）工具使用准确性和选择性执行，（iv）多步骤战略规划、错误检测和恢复，以及（v）实时动态响应能力。我们的结果表明，虽然一些架构本身表现出这些属性的子集，但有针对性的增强可以大大提高模块化代理的性能，特别是在复杂、多步骤和实时渗透测试任务中。



## **38. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

NeuronStrike：对对齐的LLM的神经元级攻击 cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2509.11864v2) [paper-pdf](https://arxiv.org/pdf/2509.11864v2)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.

摘要: 安全一致对于大型语言模型（LLM）的道德部署至关重要，指导它们避免生成有害或不道德的内容。当前的对齐技术，例如有监督的微调和来自人类反馈的强化学习，仍然很脆弱，可以被精心设计的对抗提示绕过。不幸的是，此类攻击依赖于试错，缺乏跨模型的通用性，并且受到可扩展性和可靠性的限制。   本文介绍了NeuroStrike，这是一种新颖且可推广的攻击框架，它利用了对齐技术引入的一个基本漏洞：依赖于负责检测和抑制有害输入的稀疏、专门的安全神经元。我们将NeuroStrike应用于白盒和黑盒设置：在白盒设置中，NeuroStrike通过反馈激活分析识别安全神经元，并在推理期间修剪它们以禁用安全机制。在黑匣子环境中，我们提出了第一次LLM剖析攻击，该攻击通过在开权重代理模型上训练对抗提示生成器，然后将它们部署到黑匣子和专有目标上来利用安全神经元的可移植性。我们对来自主要LLM开发商的20多个开量级LLM进行了评估。通过删除目标层中不到0.6%的神经元，NeuroStrike仅使用普通恶意提示即可实现76.9%的平均攻击成功率（ASB）。此外，Neurostrike将四种多模式LLM推广到对不安全图像输入具有100%的ASB。安全神经元在架构之间有效转移，使11个微调模型的ASB达到78.5%，5个提炼模型的ASB达到77.7%。黑匣子LLM分析攻击在包括Google Gemini系列在内的五种黑匣子型号中实现了63.7%的平均ASB。



## **39. Failures to Surface Harmful Contents in Video Large Language Models**

未能在视频大语言模型中暴露有害内容 cs.MM

12 pages, 8 figures. Accepted to AAAI 2026

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2508.10974v2) [paper-pdf](https://arxiv.org/pdf/2508.10974v2)

**Authors**: Yuxin Cao, Wei Song, Derui Wang, Jingling Xue, Jin Song Dong

**Abstract**: Video Large Language Models (VideoLLMs) are increasingly deployed on numerous critical applications, where users rely on auto-generated summaries while casually skimming the video stream. We show that this interaction hides a critical safety gap: if harmful content is embedded in a video, either as full-frame inserts or as small corner patches, state-of-the-art VideoLLMs rarely mention the harmful content in the output, despite its clear visibility to human viewers. A root-cause analysis reveals three compounding design flaws: (1) insufficient temporal coverage resulting from the sparse, uniformly spaced frame sampling used by most leading VideoLLMs, (2) spatial information loss introduced by aggressive token downsampling within sampled frames, and (3) encoder-decoder disconnection, whereby visual cues are only weakly utilized during text generation. Leveraging these insights, we craft three zero-query black-box attacks, aligning with these flaws in the processing pipeline. Our large-scale evaluation across five leading VideoLLMs shows that the harmfulness omission rate exceeds 90% in most cases. Even when harmful content is clearly present in all frames, these models consistently fail to identify it. These results underscore a fundamental vulnerability in current VideoLLMs' designs and highlight the urgent need for sampling strategies, token compression, and decoding mechanisms that guarantee semantic coverage rather than speed alone.

摘要: 视频大型语言模型（VideoLLM）越来越多地部署在许多关键应用程序上，其中用户依赖自动生成的摘要，同时随意浏览视频流。我们表明，这种交互隐藏着一个关键的安全差距：如果有害内容嵌入视频中，无论是作为全帧插入还是作为小角补丁，那么最先进的VideoLLM很少在输出中提及有害内容，尽管它对人类观众来说是清晰可见的。根本原因分析揭示了三个复合设计缺陷：（1）大多数领先的VideoLLM使用的稀疏、均匀间隔的帧采样导致的时间覆盖不足，（2）采样帧内的激进令牌下采样引入的空间信息丢失，以及（3）编码器-解码器断开连接，从而视觉线索在文本生成过程中仅被微弱地利用。利用这些见解，我们设计了三种零查询黑匣子攻击，以与处理管道中的这些缺陷保持一致。我们对五家领先的VideoLLM进行的大规模评估显示，在大多数情况下，危害性遗漏率超过90%。即使有害内容明显存在于所有帧中，这些模型仍然无法识别它。这些结果强调了当前VideoLLM设计中的一个根本漏洞，并强调了对保证语义覆盖而不仅仅是速度的采样策略、令牌压缩和解码机制的迫切需要。



## **40. Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs**

利用协同认知偏见来绕过LLC的安全性 cs.CL

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2507.22564v2) [paper-pdf](https://arxiv.org/pdf/2507.22564v2)

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.

摘要: 大型语言模型（LLM）在广泛的任务中表现出令人印象深刻的能力，但它们的安全机制仍然容易受到利用认知偏见（系统性偏离理性判断）的对抗攻击。与之前专注于即时工程或算法操纵的越狱方法不同，这项工作强调了多偏差相互作用在破坏LLM保障措施方面被忽视的力量。我们提出了CognitiveAttack，这是一种新型的红色团队框架，可以系统地利用个人和组合的认知偏见。通过集成有监督的微调和强化学习，CognitiveAttack生成嵌入优化的偏差组合的提示，有效地绕过安全协议，同时保持高攻击成功率。实验结果揭示了30种不同的LLM存在重大漏洞，特别是在开源模型中。与SOTA黑匣子方法PAP相比，CognitiveAttack的攻击成功率高得多（60.1% vs 31.6%），暴露了当前防御机制的严重局限性。这些发现凸显了多偏见相互作用是一种强大但未充分探索的攻击载体。这项工作通过连接认知科学和LLM安全性，引入了一种新颖的跨学科视角，为更强大、更人性化的人工智能系统铺平了道路。



## **41. LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge**

LLM无法可靠地判断（还吗？）：法学硕士作为法官稳健性的综合评估 cs.CR

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2506.09443v2) [paper-pdf](https://arxiv.org/pdf/2506.09443v2)

**Authors**: Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, Shouling Ji

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks, driving the development and widespread adoption of LLM-as-a-Judge systems for automated evaluation, including red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising critical concerns about their robustness and trustworthiness. Existing evaluation methods for LLM-based judges are often fragmented and lack a unified framework for comprehensive robustness assessment. Furthermore, the impact of prompt template design and model selection on judge robustness has rarely been explored, and their performance in real-world deployments remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. Specifically, RobustJudge investigates the effectiveness of 15 attack methods and 7 defense strategies across 12 models (RQ1), examines the impact of prompt template design and model selection (RQ2), and evaluates the security of real-world deployments (RQ3). Our study yields three key findings: (1) LLM-as-a-Judge systems are highly vulnerable to attacks such as PAIR and combined attacks, while defense mechanisms such as re-tokenization and LLM-based detectors can provide enhanced protection; (2) robustness varies substantially across prompt templates (up to 40%); (3) deploying RobustJudge on Alibaba's PAI platform uncovers previously undiscovered vulnerabilities. These results offer practical insights for building trustworthy LLM-as-a-Judge systems.

摘要: 大型语言模型（LLM）在不同任务中表现出了卓越的能力，推动了LLM作为法官自动评估系统的开发和广泛采用，包括红色团队和基准测试。然而，这些系统很容易受到对抗攻击，这些攻击可以操纵评估结果，从而引发对其稳健性和可信性的严重担忧。基于LLM的法官的现有评估方法往往支离破碎，缺乏全面稳健性评估的统一框架。此外，人们很少探讨即时模板设计和模型选择对判断稳健性的影响，而且它们在现实世界部署中的性能在很大程度上仍然未经验证。为了解决这些差距，我们引入了RobustJudge，这是一个全自动化和可扩展的框架，旨在系统性评估法学硕士即法官系统的稳健性。具体来说，RobustJudge调查了12个模型中15种攻击方法和7种防御策略的有效性（RJ 1），检查了即时模板设计和模型选择的影响（RJ 2），并评估现实世界部署的安全性（RJ 3）。我们的研究得出了三个关键发现：（1）LLM as-a-Judge系统极易受到PAIR和组合攻击等攻击，而重标记化和基于LLM的检测器等防御机制可以提供增强的保护;（2）不同提示模板的稳健性差异很大（高达40%）;（3）在阿里巴巴的PRI平台上部署RobustJudge发现了之前未发现的漏洞。这些结果为构建值得信赖的法学硕士作为法官系统提供了实用见解。



## **42. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

MCA-Bench：评估CAPTCHA针对基于VLM的攻击的稳健性的多模式基准 cs.CV

we update the paper supplement

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2506.05982v6) [paper-pdf](https://arxiv.org/pdf/2506.05982v6)

**Authors**: Zonglin Wu, Yule Xue, Yaoyao Feng, Xiaolong Wang, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.

摘要: 随着自动攻击技术的迅速发展，验证码仍然是针对恶意机器人的重要防御机制。然而，现有的CAPTCHA方案涵盖了多种形式--从静态扭曲文本和模糊图像到交互式点击、滑动谜题和基于逻辑的问题--但社区仍然缺乏统一的、大规模的、多模式基准来严格评估其安全稳健性。为了解决这一差距，我们引入了MCA-Bench，这是一个全面且可重复的基准测试套件，可将异类CAPTCHA类型集成到单个评估协议中。利用共享的视觉语言模型主干，我们为每个CAPTCHA类别微调专门的破解剂，实现一致的跨模式评估。大量实验表明，MCA-Bench有效地绘制了现代CAPTCHA设计在不同攻击环境下的脆弱性谱，并且至关重要地提供了挑战复杂性、交互深度和模型可解性如何相互关联的首次定量分析。基于这些发现，我们提出了三项可操作的设计原则，并确定了关键的开放挑战，为系统性CAPTCHA强化、公平的基准测试和更广泛的社区合作奠定了基础。数据集和代码可在线获取。



## **43. Chain-of-Lure: A Universal Jailbreak Attack Framework using Unconstrained Synthetic Narratives**

Chain-of-Lure：一个使用无约束合成叙述的通用越狱攻击框架 cs.CR

23 pages, 3 main figures

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2505.17519v2) [paper-pdf](https://arxiv.org/pdf/2505.17519v2)

**Authors**: Wenhan Chang, Tianqing Zhu, Yu Zhao, Shuangyong Song, Ping Xiong, Wanlei Zhou

**Abstract**: In the era of rapid generative AI development, interactions with large language models (LLMs) pose increasing risks of misuse. Prior research has primarily focused on attacks using template-based prompts and optimization-oriented methods, while overlooking the fact that LLMs possess strong unconstrained deceptive capabilities to attack other LLMs. This paper introduces a novel jailbreaking method inspired by the Chain-of-Thought mechanism. The attacker employs mission transfer to conceal harmful user intent within dialogue and generates a progressive chain of lure questions without relying on predefined templates, enabling successful jailbreaks. To further improve the attack's strength, we incorporate a helper LLM model that performs randomized narrative optimization over multi-turn interactions, enhancing the attack performance while preserving alignment with the original intent. We also propose a toxicity-based framework using third-party LLMs to evaluate harmful content and its alignment with malicious intent. Extensive experiments demonstrate that our method consistently achieves high attack success rates and elevated toxicity scores across diverse types of LLMs under black-box API settings. These findings reveal the intrinsic potential of LLMs to perform unrestricted attacks in the absence of robust alignment constraints. Our approach offers data-driven insights to inform the design of future alignment mechanisms. Finally, we propose two concrete defense strategies to support the development of safer generative models.

摘要: 在快速生成式人工智能发展的时代，与大型语言模型（LLM）的交互带来了越来越大的滥用风险。之前的研究主要集中在使用基于模板的提示和面向优化的方法的攻击上，而忽视了LLM拥有强大的不受限制的欺骗能力来攻击其他LLM这一事实。本文介绍了一种受思想链机制启发的新颖越狱方法。攻击者利用任务转移来隐藏对话中的有害用户意图，并在不依赖预定义模板的情况下生成渐进的诱饵问题链，从而实现成功越狱。为了进一步提高攻击的强度，我们引入了一个助手LLM模型，该模型在多回合交互中执行随机叙事优化，增强攻击性能，同时保持与最初意图的一致。我们还提出了一个基于毒性的框架，使用第三方LLM来评估有害内容及其与恶意意图的一致性。大量实验表明，在黑匣子API设置下，我们的方法在不同类型的LLM中始终实现了高攻击成功率和更高的毒性评分。这些发现揭示了LLM在缺乏稳健对齐约束的情况下执行无限制攻击的内在潜力。我们的方法提供数据驱动的见解，为未来对齐机制的设计提供信息。最后，我们提出了两种具体的防御策略来支持更安全的生成模型的开发。



## **44. Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models**

稳健语言模型的思想链驱动的对抗场景外推 cs.CL

19 pages, 5 figures. Accepted in AAAI 2026

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2505.17089v2) [paper-pdf](https://arxiv.org/pdf/2505.17089v2)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Ye Wang, Gang Tan, Shagufta Mehnaz

**Abstract**: Large Language Models (LLMs) exhibit impressive capabilities, but remain susceptible to a growing spectrum of safety risks, including jailbreaks, toxic content, hallucinations, and bias. Existing defenses often address only a single threat type or resort to rigid outright rejection, sacrificing user experience and failing to generalize across diverse and novel attacks. This paper introduces Adversarial Scenario Extrapolation (ASE), a novel inference-time computation framework that leverages Chain-of-Thought (CoT) reasoning to simultaneously enhance LLM robustness and seamlessness. ASE guides the LLM through a self-generative process of contemplating potential adversarial scenarios and formulating defensive strategies before generating a response to the user query. Comprehensive evaluation on four adversarial benchmarks with four latest LLMs shows that ASE achieves near-zero jailbreak attack success rates and minimal toxicity, while slashing outright rejections to <4%. ASE outperforms six state-of-the-art defenses in robustness-seamlessness trade-offs, with 92-99% accuracy on adversarial Q&A and 4-10x lower bias scores. By transforming adversarial perception into an intrinsic cognitive process, ASE sets a new paradigm for secure and natural human-AI interaction.

摘要: 大型语言模型（LLM）表现出令人印象深刻的能力，但仍然容易受到越来越多的安全风险的影响，包括越狱、有毒内容、幻觉和偏见。现有的防御系统通常只解决单一威胁类型，或者诉诸严格的彻底拒绝，牺牲用户体验，并且未能普遍适用于各种新颖的攻击。本文介绍了对抗场景外推（ASE），这是一种新型的推理时计算框架，它利用思想链（CoT）推理来同时增强LLM稳健性和无缝性。ASE指导LLM完成一个自我生成的过程，即在对用户查询做出响应之前考虑潜在的对抗场景并制定防御策略。对四种最新LLM的四种对抗基准进行的综合评估表明，ASE的越狱攻击成功率接近零，毒性最小，同时将彻底拒绝率降至< 4%。ASE在稳健性与无缝性权衡方面优于六种最先进的防御，对抗性问答的准确率为92-99%，偏见得分低4- 10倍。通过将对抗性感知转化为内在认知过程，ASE为安全、自然的人机交互设定了新范式。



## **45. Use as Many Surrogates as You Want: Selective Ensemble Attack to Unleash Transferability without Sacrificing Resource Efficiency**

使用尽可能多的代理人：选择性发起攻击以释放可转让性，而不牺牲资源效率 cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2505.12644v2) [paper-pdf](https://arxiv.org/pdf/2505.12644v2)

**Authors**: Bo Yang, Hengwei Zhang, Jindong Wang, Yuchen Ren, Chenhao Lin, Chao Shen, Zhengyu Zhao

**Abstract**: In surrogate ensemble attacks, using more surrogate models yields higher transferability but lower resource efficiency. This practical trade-off between transferability and efficiency has largely limited existing attacks despite many pre-trained models are easily accessible online. In this paper, we argue that such a trade-off is caused by an unnecessary common assumption, i.e., all models should be \textit{identical} across iterations. By lifting this assumption, we can use as many surrogates as we want to unleash transferability without sacrificing efficiency. Concretely, we propose Selective Ensemble Attack (SEA), which dynamically selects diverse models (from easily accessible pre-trained models) across iterations based on our new interpretation of decoupling within-iteration and cross-iteration model diversity. In this way, the number of within-iteration models is fixed for maintaining efficiency, while only cross-iteration model diversity is increased for higher transferability. Experiments on ImageNet demonstrate the superiority of SEA in various scenarios. For example, when dynamically selecting 4 from 20 accessible models, SEA yields 8.5% higher transferability than existing attacks under the same efficiency. The superiority of SEA also generalizes to real-world systems, such as commercial vision APIs and large vision-language models. Overall, SEA opens up the possibility of adaptively balancing transferability and efficiency according to specific resource requirements.

摘要: 在代理集成攻击中，使用更多代理模型会产生更高的可移植性，但资源效率较低。尽管许多预先训练的模型可以轻松在线访问，但可移植性和效率之间的这种实际权衡在很大程度上限制了现有的攻击。在本文中，我们认为这种权衡是由不必要的共同假设引起的，即，所有模型在迭代中都应该\textit{equivalent}。通过取消这一假设，我们可以使用尽可能多的代理人，以释放可转移性，而不牺牲效率。具体来说，我们提出了选择性集合攻击（SEA），它基于我们对迭代内脱钩和跨迭代模型多样性的新解释，在迭代中动态选择不同的模型（从易于访问的预训练模型中）。通过这种方式，迭代内模型的数量是固定的，以保持效率，而仅增加交叉迭代模型的多样性以获得更高的可移植性。ImageNet上的实验证明了SEA在各种场景下的优越性。例如，当从20个可访问模型中动态选择4个时，在相同效率下，SEA的可转移性比现有攻击高出8.5%。SEA的优势还推广到现实世界的系统，例如商业视觉API和大型视觉语言模型。总体而言，SEA开辟了根据特定资源要求自适应地平衡可转移性和效率的可能性。



## **46. FaceShield: Explainable Face Anti-Spoofing with Multimodal Large Language Models**

FaceShield：使用多模式大型语言模型的可解释面部反欺骗 cs.CV

Accepted by AAAI 2025. Hongyang Wang and Yichen Shi contribute equally. Corresponding author: Zitong Yu

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2505.09415v2) [paper-pdf](https://arxiv.org/pdf/2505.09415v2)

**Authors**: Hongyang Wang, Yichen Shi, Zhuofu Tao, Yuhao Gao, Liepiao Zhang, Xun Lin, Jun Feng, Xiaochen Yuan, Zitong Yu, Xiaochun Cao

**Abstract**: Face anti-spoofing (FAS) is crucial for protecting facial recognition systems from presentation attacks. Previous methods approached this task as a classification problem, lacking interpretability and reasoning behind the predicted results. Recently, multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and decision-making in visual tasks. However, there is currently no universal and comprehensive MLLM and dataset specifically designed for FAS task. To address this gap, we propose FaceShield, a MLLM for FAS, along with the corresponding pre-training and supervised fine-tuning (SFT) datasets, FaceShield-pre10K and FaceShield-sft45K. FaceShield is capable of determining the authenticity of faces, identifying types of spoofing attacks, providing reasoning for its judgments, and detecting attack areas. Specifically, we employ spoof-aware vision perception (SAVP) that incorporates both the original image and auxiliary information based on prior knowledge. We then use an prompt-guided vision token masking (PVTM) strategy to random mask vision tokens, thereby improving the model's generalization ability. We conducted extensive experiments on three benchmark datasets, demonstrating that FaceShield significantly outperforms previous deep learning models and general MLLMs on four FAS tasks, i.e., coarse-grained classification, fine-grained classification, reasoning, and attack localization. Our instruction datasets, protocols, and codes will be released at https://github.com/Why0912/FaceShield.

摘要: 面部反欺骗（FAA）对于保护面部识别系统免受演示攻击至关重要。之前的方法将此任务视为分类问题，缺乏预测结果背后的解释性和推理。最近，多模式大型语言模型（MLLM）在视觉任务中表现出了强大的感知、推理和决策能力。然而，目前还没有通用、全面的MLLM和专门为FAA任务设计的数据集。为了解决这一差距，我们提出了FaceShield，一个用于FAS的MLLM，以及相应的预训练和监督微调（SFT）数据集FaceShield-pre 10 K和FaceShield-sft 45 K。FaceShield能够确定人脸的真实性，识别欺骗攻击的类型，为其判断提供推理，并检测攻击区域。具体来说，我们采用欺骗感知视觉感知（SAVP），它结合了原始图像和辅助信息的基础上先验知识。然后，我们使用一个随机引导的视觉标记掩蔽（PVTM）策略来随机掩蔽视觉标记，从而提高模型的泛化能力。我们对三个基准数据集进行了广泛的实验，证明FaceShield在四个FAA任务（即粗粒度分类、细粒度分类、推理和攻击定位。我们的指令数据集、协议和代码将在https://github.com/Why0912/FaceShield上发布。



## **47. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

大型推理模型的隐藏风险：R1的安全评估 cs.CY

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2502.12659v4) [paper-pdf](https://arxiv.org/pdf/2502.12659v4)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models (LRMs), such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source reasoning models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on open LRMs is needed. (2) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (3) Safety thinking emerges in the reasoning process of LRMs, but fails frequently against adversarial attacks. (4) The thinking process in R1 models poses greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.

摘要: OpenAI-o3和DeepSeek-R1等大型推理模型（LRM）的快速发展使复杂推理相对于非推理大型语言模型（LRM）有了显着改进。然而，它们增强的功能，加上DeepSeek-R1等模型的开源访问，引发了严重的安全问题，特别是关于它们被滥用的可能性。在这项工作中，我们对这些推理模型进行了全面的安全评估，利用既定的安全基准来评估它们对安全法规的遵守性。此外，我们还调查了它们对越狱和即时注射等对抗攻击的敏感性，以评估它们在现实应用中的稳健性。通过多方面的分析，我们发现了四个关键发现：（1）开源推理模型和o3-mini模型之间在安全基准和攻击方面存在显着的安全差距，这表明需要对开放LRM做出更多的安全努力。(2)模型的推理能力越强，在回答不安全问题时可能造成的潜在危害就越大。(3)安全思维出现在LRM的推理过程中，但在对抗性攻击时经常失败。(4)R1模型中的思维过程比其最终答案带来了更大的安全问题。我们的研究为推理模型的安全性影响提供了深入的见解，并强调了进一步提高R1模型安全性以缩小差距的必要性。



## **48. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

修剪攻击图：优化隐形越狱提示生成以增强LLM内容审核 cs.CR

14 pages, 5 figures; published in EMNLP 2025 ; Code at: https://github.com/dsbuddy/GAP-LLM-Safety

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2501.18638v3) [paper-pdf](https://arxiv.org/pdf/2501.18638v3)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.

摘要: 随着大型语言模型（LLM）变得越来越普遍，确保其针对对抗性滥用的鲁棒性至关重要。本文介绍了GAP（带有修剪的攻击图）框架，这是一种生成隐形越狱提示以评估和增强LLM保障措施的高级方法。GAP通过实现互连的图结构来解决现有基于树的LLM越狱方法的局限性，该结构能够实现跨攻击路径的知识共享。我们的实验评估证明了GAP相对于现有技术的优越性，攻击成功率提高了20.8%，同时将查询成本降低了62.7%。对于攻击开放式和封闭式LLM，RAP始终优于最先进的方法，攻击成功率> 96%。此外，我们还提供了专门的变体，例如用于自动种子生成的GAP-Auto和用于多模式攻击的GAP-VLM。事实证明，由间隙生成的提示在改进内容审核系统方面非常有效，用于微调时，真阳性检测率可提高108.5%，准确率可提高183.6%。我们的实施可在https://github.com/dsbuddy/GAP-LLM-Safety上获取。



## **49. Transferability of Adversarial Attacks in Video-based MLLMs: A Cross-modal Image-to-Video Approach**

基于视频的MLLM中对抗性攻击的可转移性：跨模式图像到视频的方法 cs.CV

**SubmitDate**: 2025-11-15    [abs](http://arxiv.org/abs/2501.01042v3) [paper-pdf](https://arxiv.org/pdf/2501.01042v3)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models - a common and practical real-world scenario - remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal large language model (I-MLLM) as a surrogate model to craft adversarial video samples. Multimodal interactions and spatiotemporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. Additionally, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as a surrogate model) achieve competitive performance, with average attack success rate (AASR) of 57.98% on MSVD-QA and 58.26% on MSRVTT-QA for Zero-Shot VideoQA tasks, respectively.

摘要: 基于视频的多模式大型语言模型（V-MLLM）在视频-文本多模式任务中表现出对对抗示例的脆弱性。然而，对抗视频到未见过的模型的可移植性（一种常见且实用的现实世界场景）仍然有待探索。在本文中，我们率先研究了对抗视频样本在V-MLLM之间的可转移性。我们发现，现有的对抗攻击方法在V-MLLM的黑匣子设置中应用时面临着显着的局限性，我们将其归因于以下缺点：（1）在干扰视频特征方面缺乏一般化，（2）仅关注稀疏关键帧，（3）未能集成多模式信息。为了解决这些限制并加深对黑匣子场景中V-MLLM漏洞的了解，我们引入了图像转视频MLLM（I2 V-MLLM）攻击。在I2 V-MLLM中，我们利用基于图像的多模式大型语言模型（I-MLLM）作为代理模型来制作对抗性视频样本。多模式交互和时空信息被集成，以破坏潜在空间内的视频表示，提高对抗性可转移性。此外，还引入了扰动传播技术来处理不同的未知帧采样策略。实验结果表明，我们的方法可以生成对抗性示例，这些示例在多个视频-文本多模式任务上的不同V-MLLM之间表现出很强的可移植性。与对这些模型的白盒攻击相比，我们的黑匣子攻击（使用BLIP-2作为替代模型）实现了有竞争力的性能，对于Zero-Shot VideoQA任务，MSVD-QA的平均攻击成功率（AASB）分别为57.98%和58.26%。



## **50. What You See Is Not Always What You Get: Evaluating GPT's Comprehension of Source Code**

您所看到的并不总是您所得到的：评估GPT对源代码的理解 cs.SE

This work has been accepted at APSEC 2025

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2412.08098v3) [paper-pdf](https://arxiv.org/pdf/2412.08098v3)

**Authors**: Jiawen Wen, Bangshuo Zhu, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering tasks, including code generation and comprehension. While LLMs have shown significant potential in assisting with coding, LLMs are vulnerable to adversarial attacks. In this paper, we investigate the vulnerability of LLMs to imperceptible attacks. This class of attacks manipulate source code at the character level, which renders the changes invisible to human reviewers yet effective in misleading LLMs' behaviour. We devise these attacks into four distinct categories and analyse their impacts on code analysis and comprehension tasks. These four types of imperceptible character attacks include coding reordering, invisible coding characters, code deletions, and code homoglyphs. To assess the robustness of state-of-the-art LLMs, we present a systematic evaluation across multiple models using both perturbed and clean code snippets. Two evaluation metrics, model confidence using log probabilities of response and response correctness, are introduced. The results reveal that LLMs are susceptible to imperceptible coding perturbations, with varying degrees of degradation highlighted across different LLMs. Furthermore, we observe a consistent negative correlation between perturbation magnitude and model performance. These results highlight the urgent need for robust LLMs capable of manoeuvring behaviours under imperceptible adversarial conditions.

摘要: 最近的研究证明了大型语言模型（LLM）在软件工程任务（包括代码生成和理解）中的出色能力。虽然LLM在协助编码方面表现出了巨大的潜力，但LLM很容易受到对抗攻击。在本文中，我们研究了LLM对不可感知攻击的脆弱性。这类攻击在字符级别操纵源代码，这使得更改对人类审查者来说是不可见的，但却有效误导LLM的行为。我们将这些攻击分为四个不同的类别，并分析它们对代码分析和理解任务的影响。这四种不可感知的字符攻击包括编码重新排序、隐形编码字符、代码删除和代码同字形。为了评估最先进的LLM的稳健性，我们使用扰动和干净的代码片段对多个模型进行了系统性评估。引入了两个评估指标，即使用响应的日志概率的模型置信度和响应正确性。结果表明，LLM容易受到不可察觉的编码扰动，不同LLM之间突出显示了不同程度的退化。此外，我们观察到一个一致的扰动幅度和模型性能之间的负相关。这些结果强调了迫切需要强大的LLM能够在难以察觉的对抗条件下操纵行为。




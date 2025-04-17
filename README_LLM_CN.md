# Latest Large Language Model Attack Papers
**update at 2025-04-17 11:37:33**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails**

LLM护栏中的快速注射和越狱检测 cs.CR

12 pages, 5 figures, 6 tables

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11168v2) [paper-pdf](http://arxiv.org/pdf/2504.11168v2)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.

摘要: 大型语言模型（LLM）护栏系统旨在防止即时注入和越狱攻击。然而，他们仍然容易受到逃避技术的影响。我们演示了两种通过传统的字符注入方法和算法对抗机器学习（ML）规避技术绕过LLM提示注入和越狱检测系统的方法。通过对六种主要保护系统（包括微软的Azure Promise Shield和Meta的Promise Guard）进行测试，我们表明这两种方法都可以用来逃避检测，同时保持对抗性效用，在某些情况下实现高达100%的逃避成功。此外，我们还证明，对手可以通过利用离线白盒模型计算的单词重要性排名来提高针对黑盒目标的攻击成功率（ASB）。我们的研究结果揭示了当前LLM保护机制中的漏洞，并强调了对更坚固的护栏系统的需求。



## **2. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

软提示威胁：通过嵌入空间攻击开源LLM中的安全一致和取消学习 cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2402.09063v2) [paper-pdf](http://arxiv.org/pdf/2402.09063v2)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.

摘要: 当前对LLM对抗鲁棒性的研究重点是自然语言空间中的离散输入操纵，其可以直接转移到闭源模型。然而，这种方法忽视了开源模型的稳定发展。随着开源模型功能的进步，确保其安全性也变得越来越重要。然而，针对利用完整模型访问的开源LLM量身定制的攻击在很大程度上仍然未被探索。我们解决了这一研究空白并提出了嵌入空间攻击，该攻击直接攻击输入令牌的连续嵌入表示。我们发现，嵌入空间攻击比离散攻击或模型微调更有效地规避模型对齐并触发有害行为。此外，我们在取消学习的背景下提出了一种新颖的威胁模型，并表明嵌入空间攻击可以从多个数据集和模型中未学习的LLM中提取据称已删除的信息。我们的研究结果强调将空间攻击嵌入到开源LLM中作为重要威胁模型。触发警告：附录包含LLM生成的带有暴力和骚扰的文本。



## **3. LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in Current Benchmarks**

LLM取消学习揭示了当前基准中强于预期的核心集效应 cs.CL

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.10185v2) [paper-pdf](http://arxiv.org/pdf/2504.10185v2)

**Authors**: Soumyadeep Pal, Changsheng Wang, James Diffenderfer, Bhavya Kailkhura, Sijia Liu

**Abstract**: Large language model unlearning has become a critical challenge in ensuring safety and controlled model behavior by removing undesired data-model influences from the pretrained model while preserving general utility. Significant recent efforts have been dedicated to developing LLM unlearning benchmarks such as WMDP (Weapons of Mass Destruction Proxy) and MUSE (Machine Unlearning Six-way Evaluation), facilitating standardized unlearning performance assessment and method comparison. Despite their usefulness, we uncover for the first time a novel coreset effect within these benchmarks. Specifically, we find that LLM unlearning achieved with the original (full) forget set can be effectively maintained using a significantly smaller subset (functioning as a "coreset"), e.g., as little as 5% of the forget set, even when selected at random. This suggests that LLM unlearning in these benchmarks can be performed surprisingly easily, even in an extremely low-data regime. We demonstrate that this coreset effect remains strong, regardless of the LLM unlearning method used, such as NPO (Negative Preference Optimization) and RMU (Representation Misdirection Unlearning), the popular ones in these benchmarks. The surprisingly strong coreset effect is also robust across various data selection methods, ranging from random selection to more sophisticated heuristic approaches. We explain the coreset effect in LLM unlearning through a keyword-based perspective, showing that keywords extracted from the forget set alone contribute significantly to unlearning effectiveness and indicating that current unlearning is driven by a compact set of high-impact tokens rather than the entire dataset. We further justify the faithfulness of coreset-unlearned models along additional dimensions, such as mode connectivity and robustness to jailbreaking attacks. Codes are available at https://github.com/OPTML-Group/MU-Coreset.

摘要: 大型语言模型取消学习已成为通过从预训练模型中消除不希望的数据模型影响同时保持通用性来确保安全性和受控模型行为的一个关键挑战。最近做出了重大努力，致力于开发LLM忘记学习基准，例如WMDP（大规模杀伤性武器代理）和MUSE（机器忘记学习六路评估），促进标准化忘记学习性能评估和方法比较。尽管它们很有用，但我们首次在这些基准中发现了一种新颖的核心重置效应。具体来说，我们发现用原始（完整）忘记集实现的LLM取消学习可以使用明显较小的子集（充当“核心集”）有效地维护，例如，即使是随机选择，也只有忘记集的5%。这表明，即使在数据量极低的情况下，这些基准中的LLM取消学习也可以非常容易地执行。我们证明，无论使用何种LLM取消学习方法，例如NPO（负偏好优化）和RMU（代表误导取消学习），这种核心重置效应仍然很强，这些基准中流行的方法。令人惊讶的强烈核心集效应在各种数据选择方法中也很强大，从随机选择到更复杂的启发式方法。我们通过基于关键词的角度解释了LLM取消学习中的核心重置效应，表明仅从忘记集中提取的关键词对取消学习有效性做出了显着贡献，并表明当前的取消学习是由一组紧凑的高影响力令牌驱动的，而不是整个数据集。我们从其他维度（例如模式连接性和对越狱攻击的鲁棒性）进一步证明了未学习核心集的模型的忠实性。代码可访问https://github.com/OPTML-Group/MU-Coreset。



## **4. Entropy-Guided Watermarking for LLMs: A Test-Time Framework for Robust and Traceable Text Generation**

LLM的信息引导水印：用于稳健且可追溯的文本生成的测试时框架 cs.CL

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.12108v1) [paper-pdf](http://arxiv.org/pdf/2504.12108v1)

**Authors**: Shizhan Cai, Liang Ding, Dacheng Tao

**Abstract**: The rapid development of Large Language Models (LLMs) has intensified concerns about content traceability and potential misuse. Existing watermarking schemes for sampled text often face trade-offs between maintaining text quality and ensuring robust detection against various attacks. To address these issues, we propose a novel watermarking scheme that improves both detectability and text quality by introducing a cumulative watermark entropy threshold. Our approach is compatible with and generalizes existing sampling functions, enhancing adaptability. Experimental results across multiple LLMs show that our scheme significantly outperforms existing methods, achieving over 80\% improvements on widely-used datasets, e.g., MATH and GSM8K, while maintaining high detection accuracy.

摘要: 大型语言模型（LLM）的快速发展加剧了人们对内容可追溯性和潜在滥用的担忧。现有的采样文本水印方案经常面临保持文本质量和确保针对各种攻击的鲁棒检测之间的权衡。为了解决这些问题，我们提出了一种新颖的水印方案，通过引入累积水印信息阈值来提高可检测性和文本质量。我们的方法与现有的采样功能兼容并推广，增强了适应性。多个LLM的实验结果表明，我们的方案显着优于现有方法，在广泛使用的数据集上实现了超过80%的改进，例如MATH和GSM 8 K，同时保持高检测准确性。



## **5. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

代理安全工作台（ASB）：对基于LLM的代理中的攻击和防御进行形式化和基准化 cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2410.02644v3) [paper-pdf](http://arxiv.org/pdf/2410.02644v3)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.

摘要: 尽管基于LLM的代理在大型语言模型（LLM）的支持下可以使用外部工具和内存机制来解决复杂的现实世界任务，但它们也可能引入关键的安全漏洞。然而，现有文献并未全面评估针对基于LLM的代理的攻击和防御。为了解决这个问题，我们引入了代理安全工作台（ASB），这是一个全面的框架，旨在形式化、基准化和评估基于LLM的代理的攻击和防御，包括10种场景（例如，电子商务、自动驾驶、金融）、10个针对场景的代理、400多种工具、27种不同类型的攻击/防御方法和7个评估指标。基于ASB，我们对10种提示注入攻击、一种记忆中毒攻击、一种新颖的思想计划后门攻击、4种混合攻击以及13个LLM主干上的11种相应防御进行了基准测试。我们的基准测试结果揭示了代理操作不同阶段的关键漏洞，包括系统提示、用户提示处理、工具使用和内存检索，平均攻击成功率最高，为84.30%，但当前防御中表现出的有效性有限，揭示了社区在代理安全方面需要做的重要工作。我们还引入了一个新的指标来评估代理平衡实用性和安全性的能力。我们的代码可在https://github.com/agiresearch/ASB上找到。



## **6. Progent: Programmable Privilege Control for LLM Agents**

Progent：LLM代理的可编程特权控制 cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11703v1) [paper-pdf](http://arxiv.org/pdf/2504.11703v1)

**Authors**: Tianneng Shi, Jingxuan He, Zhun Wang, Linyu Wu, Hongwei Li, Wenbo Guo, Dawn Song

**Abstract**: LLM agents are an emerging form of AI systems where large language models (LLMs) serve as the central component, utilizing a diverse set of tools to complete user-assigned tasks. Despite their great potential, LLM agents pose significant security risks. When interacting with the external world, they may encounter malicious commands from attackers, leading to the execution of dangerous actions. A promising way to address this is by enforcing the principle of least privilege: allowing only essential actions for task completion while blocking unnecessary ones. However, achieving this is challenging, as it requires covering diverse agent scenarios while preserving both security and utility.   We introduce Progent, the first privilege control mechanism for LLM agents. At its core is a domain-specific language for flexibly expressing privilege control policies applied during agent execution. These policies provide fine-grained constraints over tool calls, deciding when tool calls are permissible and specifying fallbacks if they are not. This enables agent developers and users to craft suitable policies for their specific use cases and enforce them deterministically to guarantee security. Thanks to its modular design, integrating Progent does not alter agent internals and requires only minimal changes to agent implementation, enhancing its practicality and potential for widespread adoption. To automate policy writing, we leverage LLMs to generate policies based on user queries, which are then updated dynamically for improved security and utility. Our extensive evaluation shows that it enables strong security while preserving high utility across three distinct scenarios or benchmarks: AgentDojo, ASB, and AgentPoison. Furthermore, we perform an in-depth analysis, showcasing the effectiveness of its core components and the resilience of its automated policy generation against adaptive attacks.

摘要: LLM代理是人工智能系统的一种新兴形式，其中大型语言模型（LLM）作为中心组件，利用一组不同的工具来完成用户分配的任务。尽管LLM代理潜力巨大，但仍构成重大安全风险。在与外部世界互动时，他们可能会遇到攻击者的恶意命令，导致执行危险动作。解决这个问题的一个有希望的方法是执行最小特权原则：仅允许执行完成任务的必要动作，同时阻止不必要的动作。然而，实现这一点具有挑战性，因为它需要覆盖不同的代理场景，同时保持安全性和实用性。   我们引入Progent，这是LLM代理的第一个特权控制机制。其核心是一种特定于领域的语言，用于灵活表达代理执行期间应用的特权控制策略。这些策略为工具调用提供了细粒度的约束，决定何时允许工具调用，并在不允许时指定后备。这使代理开发人员和用户能够为他们的特定用例制定合适的策略，并确定性地实施这些策略以保证安全性。由于其模块化设计，集成Progent不会改变代理的内部结构，只需要对代理的实现进行最小的更改，从而增强了其实用性和广泛采用的潜力。为了自动化策略编写，我们利用LLM根据用户查询生成策略，然后动态更新以提高安全性和实用性。我们的广泛评估表明，它可以实现强大的安全性，同时在三个不同的场景或基准测试中保持高实用性：AgentDojo，ASB和AgentPoison。此外，我们还进行了深入的分析，展示了其核心组件的有效性以及自动化策略生成针对自适应攻击的弹性。



## **7. Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' "Typo" Correction**

通过LLM辅助频谱图的“错别字”纠正，使对噪音键盘的声学侧通道攻击变得可行 cs.CR

Length: 13 pages Figures: 5 figures Tables: 7 tables Keywords:  Acoustic side-channel attacks, machine learning, Visual Transformers, Large  Language Models (LLMs), security Conference: Accepted at the 19th USENIX WOOT  Conference on Offensive Technologies (WOOT '25). Licensing: This paper is  submitted under the CC BY Creative Commons Attribution license. arXiv admin  note: text overlap with arXiv:2502.09782

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11622v1) [paper-pdf](http://arxiv.org/pdf/2504.11622v1)

**Authors**: Seyyed Ali Ayati, Jin Hyun Park, Yichen Cai, Marcus Botacin

**Abstract**: The large integration of microphones into devices increases the opportunities for Acoustic Side-Channel Attacks (ASCAs), as these can be used to capture keystrokes' audio signals that might reveal sensitive information. However, the current State-Of-The-Art (SOTA) models for ASCAs, including Convolutional Neural Networks (CNNs) and hybrid models, such as CoAtNet, still exhibit limited robustness under realistic noisy conditions. Solving this problem requires either: (i) an increased model's capacity to infer contextual information from longer sequences, allowing the model to learn that an initially noisily typed word is the same as a futurely collected non-noisy word, or (ii) an approach to fix misidentified information from the contexts, as one does not type random words, but the ones that best fit the conversation context. In this paper, we demonstrate that both strategies are viable and complementary solutions for making ASCAs practical. We observed that no existing solution leverages advanced transformer architectures' power for these tasks and propose that: (i) Visual Transformers (VTs) are the candidate solutions for capturing long-term contextual information and (ii) transformer-powered Large Language Models (LLMs) are the candidate solutions to fix the ``typos'' (mispredictions) the model might make. Thus, we here present the first-of-its-kind approach that integrates VTs and LLMs for ASCAs.   We first show that VTs achieve SOTA performance in classifying keystrokes when compared to the previous CNN benchmark. Second, we demonstrate that LLMs can mitigate the impact of real-world noise. Evaluations on the natural sentences revealed that: (i) incorporating LLMs (e.g., GPT-4o) in our ASCA pipeline boosts the performance of error-correction tasks; and (ii) the comparable performance can be attained by a lightweight, fine-tuned smaller LLM (67 times smaller than GPT-4o), using...

摘要: 麦克风大量集成到设备中增加了声学侧道攻击（ASCA）的机会，因为这些攻击可用于捕获可能泄露敏感信息的麦克风音频信号。然而，当前ASCA的最新技术水平（SOTA）模型（包括卷积神经网络（CNN）和混合模型（例如CoAtNet））在现实噪音条件下仍然表现出有限的鲁棒性。解决这个问题需要：（i）提高模型从更长的序列中推断上下文信息的能力，允许模型学习最初输入的有噪音的单词与未来收集的无噪音单词相同，或者（ii）修复来自上下文的错误识别信息的方法，因为输入的不是随机单词，而是最适合对话上下文的单词。在本文中，我们证明了这两种策略都是使ASCA实用的可行且相辅相成的解决方案。我们观察到，没有现有的解决方案利用高级转换器架构的能力来完成这些任务，并建议：（i）视觉转换器（VT）是捕获长期上下文信息的候选解决方案，（ii）转换器驱动的大型语言模型（LLM）是修复模型可能造成的“拼写错误”（预测错误）的候选解决方案。因此，我们在这里介绍了一种首创的方法，该方法将VT和LLM集成到ASCA中。   我们首先表明，与之前的CNN基准相比，VT在分类击键方面实现了SOTA性能。其次，我们证明LLM可以减轻现实世界噪音的影响。对自然句子的评估显示：（i）纳入LLM（例如，我们的ASCA管道中的GPT-4o）提高了错误纠正任务的性能;并且（ii）通过轻量级、微调的较小LLM（比GPT-4o小67倍）可以获得相当的性能，使用.



## **8. Lateral Phishing With Large Language Models: A Large Organization Comparative Study**

大型语言模型的横向网络钓鱼：大型组织比较研究 cs.CR

Accepted for publication in IEEE Access. This version includes  revisions following peer review

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2401.09727v2) [paper-pdf](http://arxiv.org/pdf/2401.09727v2)

**Authors**: Mazal Bethany, Athanasios Galiopoulos, Emet Bethany, Mohammad Bahrami Karkevandi, Nicole Beebe, Nishant Vishwamitra, Peyman Najafirad

**Abstract**: The emergence of Large Language Models (LLMs) has heightened the threat of phishing emails by enabling the generation of highly targeted, personalized, and automated attacks. Traditionally, many phishing emails have been characterized by typos, errors, and poor language. These errors can be mitigated by LLMs, potentially lowering the barrier for attackers. Despite this, there is a lack of large-scale studies comparing the effectiveness of LLM-generated lateral phishing emails to those crafted by humans. Current literature does not adequately address the comparative effectiveness of LLM and human-generated lateral phishing emails in a real-world, large-scale organizational setting, especially considering the potential for LLMs to generate more convincing and error-free phishing content. To address this gap, we conducted a pioneering study within a large university, targeting its workforce of approximately 9,000 individuals including faculty, staff, administrators, and student workers. Our results indicate that LLM-generated lateral phishing emails are as effective as those written by communications professionals, emphasizing the critical threat posed by LLMs in leading phishing campaigns. We break down the results of the overall phishing experiment, comparing vulnerability between departments and job roles. Furthermore, to gather qualitative data, we administered a detailed questionnaire, revealing insights into the reasons and motivations behind vulnerable employee's actions. This study contributes to the understanding of cyber security threats in educational institutions and provides a comprehensive comparison of LLM and human-generated phishing emails' effectiveness, considering the potential for LLMs to generate more convincing content. The findings highlight the need for enhanced user education and system defenses to mitigate the growing threat of AI-powered phishing attacks.

摘要: 大型语言模型（LLM）的出现通过生成高度针对性、个性化和自动化的攻击，加剧了网络钓鱼电子邮件的威胁。传统上，许多网络钓鱼电子邮件的特点是拼写错误、错误和语言拙劣。这些错误可以通过LLM来缓解，从而可能降低攻击者的障碍。尽管如此，缺乏大规模研究将LLM生成的横向网络钓鱼电子邮件与人类制作的横向网络钓鱼电子邮件的有效性进行比较。当前的文献没有充分解决LLM和人类生成的横向网络钓鱼电子邮件在现实世界、大规模组织环境中的比较有效性，特别是考虑到LLM生成更令人信服且无错误的网络钓鱼内容的潜力。为了解决这一差距，我们在一所大型大学内进行了一项开创性研究，目标是其约9，000名员工，包括教职员工、管理人员和学生工作者。我们的结果表明，LLM生成的横向网络钓鱼电子邮件与通信专业人士撰写的电子邮件一样有效，强调了LLM在领先的网络钓鱼活动中构成的严重威胁。我们分解了整个网络钓鱼实验的结果，比较了部门和工作角色之间的脆弱性。此外，为了收集定性数据，我们进行了一份详细的调查问卷，揭示了对弱势员工行为背后的原因和动机的见解。这项研究有助于了解教育机构的网络安全威胁，并对LLM和人类生成的网络钓鱼电子邮件的有效性进行了全面比较，同时考虑到LLM生成更令人信服的内容的潜力。研究结果凸显了加强用户教育和系统防御的必要性，以减轻人工智能驱动的网络钓鱼攻击日益严重的威胁。



## **9. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2410.02240v5) [paper-pdf](http://arxiv.org/pdf/2410.02240v5)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.

摘要: 部署在敏感环境中的基于深度神经网络的系统很容易受到对抗攻击。不受限制的对抗攻击通常操纵图像的语义内容（例如，颜色或纹理）来创建既有效又逼真的对抗示例。最近的作品利用扩散倒置过程将图像映射到潜在空间，其中通过引入扰动来操纵高级语义。然而，它们通常会导致去噪输出中出现严重的语义扭曲，并且效率低下。在这项研究中，我们提出了一种名为语义一致的无限制对抗攻击（SCA）的新型框架，该框架采用倒置方法来提取编辑友好的噪音图，并利用多模式大型语言模型（MLLM）在整个过程中提供语义指导。在MLLM提供丰富的语义信息的情况下，我们使用一系列编辑友好的噪音图来执行每一步的DDPM去噪过程，并利用DeliverSolver ++加速这一过程，实现具有语义一致性的高效采样。与现有方法相比，我们的框架能够高效生成表现出最小可辨别的语义变化的对抗性示例。因此，我们首次引入语义一致的对抗示例（SCAE）。大量的实验和可视化已经证明了SCA的高效率，特别是平均比最先进的攻击快12倍。本文的研究可以进一步引起人们对多媒体信息安全的关注。



## **10. The Obvious Invisible Threat: LLM-Powered GUI Agents' Vulnerability to Fine-Print Injections**

显而易见的不可见威胁：LLM-Powered GUI代理对Fine-Print注入的脆弱性 cs.HC

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11281v1) [paper-pdf](http://arxiv.org/pdf/2504.11281v1)

**Authors**: Chaoran Chen, Zhiping Zhang, Bingcan Guo, Shang Ma, Ibrahim Khalilov, Simret A Gebreegziabher, Yanfang Ye, Ziang Xiao, Yaxing Yao, Tianshi Li, Toby Jia-Jun Li

**Abstract**: A Large Language Model (LLM) powered GUI agent is a specialized autonomous system that performs tasks on the user's behalf according to high-level instructions. It does so by perceiving and interpreting the graphical user interfaces (GUIs) of relevant apps, often visually, inferring necessary sequences of actions, and then interacting with GUIs by executing the actions such as clicking, typing, and tapping. To complete real-world tasks, such as filling forms or booking services, GUI agents often need to process and act on sensitive user data. However, this autonomy introduces new privacy and security risks. Adversaries can inject malicious content into the GUIs that alters agent behaviors or induces unintended disclosures of private information. These attacks often exploit the discrepancy between visual saliency for agents and human users, or the agent's limited ability to detect violations of contextual integrity in task automation. In this paper, we characterized six types of such attacks, and conducted an experimental study to test these attacks with six state-of-the-art GUI agents, 234 adversarial webpages, and 39 human participants. Our findings suggest that GUI agents are highly vulnerable, particularly to contextually embedded threats. Moreover, human users are also susceptible to many of these attacks, indicating that simple human oversight may not reliably prevent failures. This misalignment highlights the need for privacy-aware agent design. We propose practical defense strategies to inform the development of safer and more reliable GUI agents.

摘要: 由大型语言模型（LLM）驱动的图形用户界面代理是一个专门的自治系统，根据高级指令代表用户执行任务。它通过感知和解释相关应用程序的图形用户界面（GUIs）（通常是视觉上的），推断必要的操作序列，然后通过执行单击、打字和点击等操作与GUIs交互来实现这一目标。为了完成现实世界的任务，例如填写表格或预订服务，图形用户界面代理通常需要处理和处理敏感用户数据。然而，这种自主性带来了新的隐私和安全风险。对手可以将恶意内容注入图形用户界面，从而改变代理行为或导致私人信息的意外泄露。这些攻击通常利用代理和人类用户的视觉显著性之间的差异，或者代理检测任务自动化中上下文完整性违规的能力有限。在本文中，我们描述了六种类型的此类攻击，并进行了一项实验研究，使用六个最先进的图形用户界面代理、234个对抗性网页和39名人类参与者来测试这些攻击。我们的研究结果表明，图形用户界面代理非常容易受到攻击，特别是对于上下文嵌入式威胁。此外，人类用户也容易受到许多此类攻击，这表明简单的人类监督可能无法可靠地防止故障。这种错位凸显了隐私感知代理设计的必要性。我们提出了实用的防御策略，为开发更安全、更可靠的图形用户界面代理提供信息。



## **11. Exploring Backdoor Attack and Defense for LLM-empowered Recommendations**

探索LLM授权建议的后门攻击和防御 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11182v1) [paper-pdf](http://arxiv.org/pdf/2504.11182v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner.

摘要: 大型语言模型（LLM）与推荐系统（RecSys）的融合极大地提高了个性化推荐并引起了广泛关注。尽管取得了令人印象深刻的进展，但基于LLM的RecSys抵御后门攻击的安全性在很大程度上仍然没有得到充分的探索。在本文中，我们提出了一个新问题：具有特定触发器的后门是否会被注入到基于LLM的Recsys中，从而导致当后门触发器附加到项目标题时推荐响应的操纵？为了调查基于LLM的RecSys在后门攻击下的漏洞，我们提出了一种新的攻击框架，称为RecSys后门注入中毒（BadRec）。BadRec通过触发器扰乱这些物品的标题，并雇用几名虚假用户与这些物品互动，有效地毒害了训练集，并为基于LLM的RecSys注入后门。全面的实验表明，仅用对抗性示例毒害1%的训练数据就足以成功植入后门，从而能够操纵推荐。为了进一步减轻此类安全威胁，我们提出了一种名为毒药扫描仪（P-Scanner）的通用防御策略。具体来说，我们引入了基于LLM的毒物扫描仪，通过利用LLM强大的语言理解能力和丰富的知识来检测有毒物品。触发增强代理被用来生成不同的合成触发器，以引导中毒扫描器学习中毒物品检测任务的特定于领域的知识。在三个真实数据集上的大量实验验证了所提出的P-Scanner的有效性。



## **12. QAVA: Query-Agnostic Visual Attack to Large Vision-Language Models**

QAVA：对大型视觉语言模型的查询不可知视觉攻击 cs.CV

Accepted by NAACL 2025 main

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11038v1) [paper-pdf](http://arxiv.org/pdf/2504.11038v1)

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Yu Wang

**Abstract**: In typical multimodal tasks, such as Visual Question Answering (VQA), adversarial attacks targeting a specific image and question can lead large vision-language models (LVLMs) to provide incorrect answers. However, it is common for a single image to be associated with multiple questions, and LVLMs may still answer other questions correctly even for an adversarial image attacked by a specific question. To address this, we introduce the query-agnostic visual attack (QAVA), which aims to create robust adversarial examples that generate incorrect responses to unspecified and unknown questions. Compared to traditional adversarial attacks focused on specific images and questions, QAVA significantly enhances the effectiveness and efficiency of attacks on images when the question is unknown, achieving performance comparable to attacks on known target questions. Our research broadens the scope of visual adversarial attacks on LVLMs in practical settings, uncovering previously overlooked vulnerabilities, particularly in the context of visual adversarial threats. The code is available at https://github.com/btzyd/qava.

摘要: 在典型的多模式任务中，例如视觉问题解答（VQA），针对特定图像和问题的对抗攻击可能会导致大型视觉语言模型（LVLM）提供错误的答案。然而，单个图像与多个问题关联是常见的，即使对于受到特定问题攻击的对抗图像，LVLM仍然可以正确回答其他问题。为了解决这个问题，我们引入了查询不可知视觉攻击（QAVA），其目的是创建强大的对抗性示例，这些示例会对未指定和未知的问题生成错误的响应。与针对特定图像和问题的传统对抗攻击相比，QAVA显着增强了问题未知时图像攻击的有效性和效率，实现了与针对已知目标问题的攻击相当的性能。我们的研究扩大了实际环境中对LVLM的视觉对抗攻击的范围，揭示了以前被忽视的漏洞，特别是在视觉对抗威胁的背景下。该代码可在https://github.com/btzyd/qava上获取。



## **13. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

迈向智能和安全的云：大型语言模型增强主动防御 cs.CR

7 pages; In submission

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2412.21051v2) [paper-pdf](http://arxiv.org/pdf/2412.21051v2)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Yuyu Zhao

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided a large number of benefits in daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks. Recent advancements in generative foundation models (GFMs), particularly in the large language models (LLMs), offer promising solutions for security intelligence. By exploiting the powerful abilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel proactive defense architecture that defeats various threats in a proactive manner. LLM-PD can efficiently make a decision through comprehensive data analysis and sequential reasoning, as well as dynamically creating and deploying actionable defense mechanisms on the target cloud. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. The experimental results demonstrate its remarkable ability in terms of defense effectiveness and efficiency, particularly highlighting an outstanding success rate when compared with other existing methods.

摘要: 云计算技术的快速发展和云应用程序数量的不断增加为日常生活带来了大量好处。然而，不同组件的多样性和复杂性对云安全构成了重大挑战，特别是在处理复杂和高级的网络攻击时。生成式基础模型（GFM）的最新进展，特别是大型语言模型（LLM），为安全智能提供了有前途的解决方案。通过利用语言理解、数据分析、任务推理、行动规划和代码生成方面的强大能力，我们提出了LLM-PD，这是一种新型的主动防御架构，可以以主动的方式击败各种威胁。LLM-PD可以通过全面的数据分析和顺序推理，以及在目标云上动态创建和部署可操作的防御机制来有效地做出决策。此外，它可以根据从之前的交互中学到的经验灵活地自我进化，并在无需额外训练的情况下适应新的攻击场景。实验结果证明了其在防御有效性和效率方面的出色能力，特别是与其他现有方法相比具有出色的成功率。



## **14. Adversarial Prompt Distillation for Vision-Language Models**

视觉语言模型的对抗性即时蒸馏 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2411.15244v2) [paper-pdf](http://arxiv.org/pdf/2411.15244v2)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.

摘要: 对比图像预训练（CLIP）等大型预训练视觉语言模型（VLM）已被证明容易受到对抗攻击，这引发了人们对其在自动驾驶和医疗诊断等安全关键应用中部署的担忧。对抗性提示调整（APT）是对预训练的VLM进行鲁棒化的一种有希望的方法，它在提示调整的过程中应用对抗性训练。然而，现有的APT方法大多是单模式方法，仅为视觉或文本模式设计提示，从而限制了其稳健性或清晰准确性的有效性。在这项工作中，我们提出了对抗性提示蒸馏（APT），这是一个双峰知识蒸馏框架，通过将APT与多模式知识转移集成来增强APT。APT优化视觉和文本模式的提示，同时从干净的预培训教师CLIP模型中提取知识。对多个基准数据集的广泛实验证明了我们的APT方法在对抗稳健性和精确性方面优于当前最先进的APT方法。APT的有效性也验证了使用非稳健教师来提高微调后的VLM的通用性和稳健性的可能性。



## **15. The Jailbreak Tax: How Useful are Your Jailbreak Outputs?**

越狱税：你的越狱输出有多有用？ cs.LG

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10694v1) [paper-pdf](http://arxiv.org/pdf/2504.10694v1)

**Authors**: Kristina Nikolić, Luze Sun, Jie Zhang, Florian Tramèr

**Abstract**: Jailbreak attacks bypass the guardrails of large language models to produce harmful outputs. In this paper, we ask whether the model outputs produced by existing jailbreaks are actually useful. For example, when jailbreaking a model to give instructions for building a bomb, does the jailbreak yield good instructions? Since the utility of most unsafe answers (e.g., bomb instructions) is hard to evaluate rigorously, we build new jailbreak evaluation sets with known ground truth answers, by aligning models to refuse questions related to benign and easy-to-evaluate topics (e.g., biology or math). Our evaluation of eight representative jailbreaks across five utility benchmarks reveals a consistent drop in model utility in jailbroken responses, which we term the jailbreak tax. For example, while all jailbreaks we tested bypass guardrails in models aligned to refuse to answer math, this comes at the expense of a drop of up to 92% in accuracy. Overall, our work proposes the jailbreak tax as a new important metric in AI safety, and introduces benchmarks to evaluate existing and future jailbreaks. We make the benchmark available at https://github.com/ethz-spylab/jailbreak-tax

摘要: 越狱攻击绕过大型语言模型的护栏，产生有害的输出。在本文中，我们询问现有越狱产生的模型输出是否真正有用。例如，当越狱模型以给出制造炸弹的指令时，越狱是否会产生良好的指令？由于大多数不安全答案（例如，炸弹指令）很难严格评估，我们通过调整模型来拒绝与良性且易于评估的主题相关的问题，使用已知的地面真相答案来构建新的越狱评估集（例如，生物学或数学）。我们对五个公用事业基准中八个代表性越狱的评估显示，越狱响应（我们将其称为越狱税）中的模型效用持续下降。例如，虽然我们测试的所有越狱都是在拒绝回答数学问题的模型中绕过护栏，但这是以准确性下降高达92%为代价的。总体而言，我们的工作提出将越狱税作为人工智能安全的新的重要指标，并引入了评估现有和未来越狱的基准。我们在https://github.com/ethz-spylab/jailbreak-tax上提供基准测试



## **16. Look Before You Leap: Enhancing Attention and Vigilance Regarding Harmful Content with GuidelineLLM**

三思而后行：通过GuidelineLLM提高对有害内容的关注和警惕 cs.CL

AAAI 2025

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2412.10423v2) [paper-pdf](http://arxiv.org/pdf/2412.10423v2)

**Authors**: Shaoqing Zhang, Zhuosheng Zhang, Kehai Chen, Rongxiang Weng, Muyun Yang, Tiejun Zhao, Min Zhang

**Abstract**: Despite being empowered with alignment mechanisms, large language models (LLMs) are increasingly vulnerable to emerging jailbreak attacks that can compromise their alignment mechanisms. This vulnerability poses significant risks to real-world applications. Existing work faces challenges in both training efficiency and generalization capabilities (i.e., Reinforcement Learning from Human Feedback and Red-Teaming). Developing effective strategies to enable LLMs to resist continuously evolving jailbreak attempts represents a significant challenge. To address this challenge, we propose a novel defensive paradigm called GuidelineLLM, which assists LLMs in recognizing queries that may have harmful content. Before LLMs respond to a query, GuidelineLLM first identifies potential risks associated with the query, summarizes these risks into guideline suggestions, and then feeds these guidelines to the responding LLMs. Importantly, our approach eliminates the necessity for additional safety fine-tuning of the LLMs themselves; only the GuidelineLLM requires fine-tuning. This characteristic enhances the general applicability of GuidelineLLM across various LLMs. Experimental results demonstrate that GuidelineLLM can significantly reduce the attack success rate (ASR) against LLM (an average reduction of 34.17\% ASR) while maintaining the usefulness of LLM in handling benign queries. The code is available at https://github.com/sqzhang-lazy/GuidelineLLM.

摘要: 尽管拥有对齐机制，大型语言模型（LLM）仍越来越容易受到新出现的越狱攻击的影响，这些攻击可能会损害其对齐机制。此漏洞对现实世界的应用程序构成了重大风险。现有工作在培训效率和概括能力方面面临挑战（即，来自人类反馈和红色团队的强化学习）。制定有效的策略以使法学硕士能够抵御不断变化的越狱企图是一项重大挑战。为了应对这一挑战，我们提出了一种名为GuidelineLLM的新型防御范式，它帮助LLM识别可能包含有害内容的查询。在LLM响应查询之前，GuidelineLLM首先识别与查询相关的潜在风险，将这些风险总结为指南建议，然后将这些指南提供给响应的LLM。重要的是，我们的方法消除了对LLM本身进行额外安全微调的必要性;只有GuidelineLLM需要微调。该特征增强了GuidelineLLM在各种LLM中的普遍适用性。实验结果表明，GuidelineLLM可以显着降低针对LLM的攻击成功率（ASB）（平均降低34.17%ASB），同时保持LLM处理良性查询的有用性。该代码可在https://github.com/sqzhang-lazy/GuidelineLLM上获取。



## **17. Using Large Language Models for Template Detection from Security Event Logs**

使用大型语言模型从安全事件收件箱进行模板检测 cs.CR

Accepted for publication in International Journal of Information  Security

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2409.05045v3) [paper-pdf](http://arxiv.org/pdf/2409.05045v3)

**Authors**: Risto Vaarandi, Hayretdin Bahsi

**Abstract**: In modern IT systems and computer networks, real-time and offline event log analysis is a crucial part of cyber security monitoring. In particular, event log analysis techniques are essential for the timely detection of cyber attacks and for assisting security experts with the analysis of past security incidents. The detection of line patterns or templates from unstructured textual event logs has been identified as an important task of event log analysis since detected templates represent event types in the event log and prepare the logs for downstream online or offline security monitoring tasks. During the last two decades, a number of template mining algorithms have been proposed. However, many proposed algorithms rely on traditional data mining techniques, and the usage of Large Language Models (LLMs) has received less attention so far. Also, most approaches that harness LLMs are supervised, and unsupervised LLM-based template mining remains an understudied area. The current paper addresses this research gap and investigates the application of LLMs for unsupervised detection of templates from unstructured security event logs.

摘要: 在现代IT系统和计算机网络中，实时和离线事件日志分析是网络安全监控的重要组成部分。特别是，事件日志分析技术对于及时检测网络攻击和协助安全专家分析过去的安全事件至关重要。从非结构化文本事件日志中检测线模式或模板已被确定为事件日志分析的重要任务，因为检测到的模板代表事件日志中的事件类型，并为下游在线或离线安全监控任务准备日志。在过去的二十年里，人们提出了许多模板挖掘算法。然而，许多提出的算法依赖于传统的数据挖掘技术，并且到目前为止，大型语言模型（LLM）的使用受到的关注较少。此外，大多数利用LLM的方法都是有监督的，而无监督的基于LLM的模板挖掘仍然是一个研究不足的领域。当前论文解决了这一研究空白，并研究了LLM在无监督检测非结构化安全事件日志模板中的应用。



## **18. Benchmarking Practices in LLM-driven Offensive Security: Testbeds, Metrics, and Experiment Design**

法学硕士驱动的攻击性安全中的基准实践：测试床、工作组和实验设计 cs.CR

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10112v1) [paper-pdf](http://arxiv.org/pdf/2504.10112v1)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: Large Language Models (LLMs) have emerged as a powerful approach for driving offensive penetration-testing tooling. This paper analyzes the methodology and benchmarking practices used for evaluating Large Language Model (LLM)-driven attacks, focusing on offensive uses of LLMs in cybersecurity. We review 16 research papers detailing 15 prototypes and their respective testbeds.   We detail our findings and provide actionable recommendations for future research, emphasizing the importance of extending existing testbeds, creating baselines, and including comprehensive metrics and qualitative analysis. We also note the distinction between security research and practice, suggesting that CTF-based challenges may not fully represent real-world penetration testing scenarios.

摘要: 大型语言模型（LLM）已成为驱动攻击性渗透测试工具的强大方法。本文分析了用于评估大型语言模型（LLM）驱动的攻击的方法论和基准实践，重点关注LLM在网络安全中的攻击性使用。我们回顾了16篇研究论文，详细介绍了15个原型及其各自的测试平台。   我们详细介绍了我们的调查结果，并为未来的研究提供了可操作的建议，强调了扩展现有测试平台、创建基线以及包括全面指标和定性分析的重要性。我们还注意到安全研究和实践之间的区别，这表明基于CTF的挑战可能无法完全代表真实世界的渗透测试场景。



## **19. From Vulnerabilities to Remediation: A Systematic Literature Review of LLMs in Code Security**

从漏洞到补救：代码安全领域LLM的系统文献评论 cs.CR

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2412.15004v3) [paper-pdf](http://arxiv.org/pdf/2412.15004v3)

**Authors**: Enna Basic, Alberto Giaretta

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for automating various programming tasks, including security-related ones, such as detecting and fixing vulnerabilities. Despite their promising capabilities, when required to produce or modify pre-existing code, LLMs could introduce vulnerabilities unbeknown to the programmer. When analyzing code, they could miss clear vulnerabilities or signal nonexistent ones. In this Systematic Literature Review (SLR), we aim to investigate both the security benefits and potential drawbacks of using LLMs for a variety of code-related tasks. In particular, first we focus on the types of vulnerabilities that could be introduced by LLMs, when used for producing code. Second, we analyze the capabilities of LLMs to detect and fix vulnerabilities, in any given code, and how the prompting strategy of choice impacts their performance in these two tasks. Last, we provide an in-depth analysis on how data poisoning attacks on LLMs can impact performance in the aforementioned tasks.

摘要: 大型语言模型（LLM）已成为自动化各种编程任务的强大工具，包括与安全相关的任务，例如检测和修复漏洞。尽管LLM的功能很有希望，但当需要生成或修改预先存在的代码时，LLM可能会引入程序员不知道的漏洞。分析代码时，他们可能会错过明显的漏洞或发出不存在的漏洞。在本系统文献评论（SLR）中，我们的目标是研究使用LLM执行各种代码相关任务的安全益处和潜在缺陷。特别是，首先我们关注LLM在用于生成代码时可能引入的漏洞类型。其次，我们分析LLM在任何给定代码中检测和修复漏洞的能力，以及选择的提示策略如何影响其在这两项任务中的性能。最后，我们深入分析了对LLM的数据中毒攻击如何影响上述任务的性能。



## **20. Do We Really Need Curated Malicious Data for Safety Alignment in Multi-modal Large Language Models?**

在多模态大型语言模型中，我们真的需要精心策划的恶意数据来进行安全对齐吗？ cs.CR

Accepted to CVPR 2025, codes in process

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10000v1) [paper-pdf](http://arxiv.org/pdf/2504.10000v1)

**Authors**: Yanbo Wang, Jiyang Guan, Jian Liang, Ran He

**Abstract**: Multi-modal large language models (MLLMs) have made significant progress, yet their safety alignment remains limited. Typically, current open-source MLLMs rely on the alignment inherited from their language module to avoid harmful generations. However, the lack of safety measures specifically designed for multi-modal inputs creates an alignment gap, leaving MLLMs vulnerable to vision-domain attacks such as typographic manipulation. Current methods utilize a carefully designed safety dataset to enhance model defense capability, while the specific knowledge or patterns acquired from the high-quality dataset remain unclear. Through comparison experiments, we find that the alignment gap primarily arises from data distribution biases, while image content, response quality, or the contrastive behavior of the dataset makes little contribution to boosting multi-modal safety. To further investigate this and identify the key factors in improving MLLM safety, we propose finetuning MLLMs on a small set of benign instruct-following data with responses replaced by simple, clear rejection sentences. Experiments show that, without the need for labor-intensive collection of high-quality malicious data, model safety can still be significantly improved, as long as a specific fraction of rejection data exists in the finetuning set, indicating the security alignment is not lost but rather obscured during multi-modal pretraining or instruction finetuning. Simply correcting the underlying data bias could narrow the safety gap in the vision domain.

摘要: 多模式大型语言模型（MLLM）已经取得了重大进展，但其安全性一致性仍然有限。通常，当前的开源MLLM依赖于从其语言模块继承的对齐来避免有害的世代。然而，缺乏专门为多模式输入设计的安全措施造成了对齐差距，使MLLM容易受到印刷操纵等视觉域攻击。当前的方法利用精心设计的安全数据集来增强模型防御能力，而从高质量数据集获取的具体知识或模式仍然不清楚。通过比较实验，我们发现对齐差距主要源于数据分布偏差，而图像内容、响应质量或数据集的对比行为对提高多模式安全性贡献不大。为了进一步研究这一点并确定提高MLLM安全性的关键因素，我们建议对一小组良性预算跟踪数据进行微调MLLM，并用简单、明确的拒绝句取代响应。实验表明，在不需要劳动密集型收集高质量恶意数据的情况下，只要微调集中存在特定比例的拒绝数据，模型安全性仍然可以显着提高，这表明安全对齐并没有丢失，而是在多模式预训练或指令微调期间被掩盖。简单地纠正潜在的数据偏见就可以缩小视觉领域的安全差距。



## **21. StruPhantom: Evolutionary Injection Attacks on Black-Box Tabular Agents Powered by Large Language Models**

StruPhantom：对由大型语言模型支持的黑盒表代理的进化注入攻击 cs.CR

Work in Progress

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.09841v1) [paper-pdf](http://arxiv.org/pdf/2504.09841v1)

**Authors**: Yang Feng, Xudong Pan

**Abstract**: The proliferation of autonomous agents powered by large language models (LLMs) has revolutionized popular business applications dealing with tabular data, i.e., tabular agents. Although LLMs are observed to be vulnerable against prompt injection attacks from external data sources, tabular agents impose strict data formats and predefined rules on the attacker's payload, which are ineffective unless the agent navigates multiple layers of structural data to incorporate the payload. To address the challenge, we present a novel attack termed StruPhantom which specifically targets black-box LLM-powered tabular agents. Our attack designs an evolutionary optimization procedure which continually refines attack payloads via the proposed constrained Monte Carlo Tree Search augmented by an off-topic evaluator. StruPhantom helps systematically explore and exploit the weaknesses of target applications to achieve goal hijacking. Our evaluation validates the effectiveness of StruPhantom across various LLM-based agents, including those on real-world platforms, and attack scenarios. Our attack achieves over 50% higher success rates than baselines in enforcing the application's response to contain phishing links or malicious codes.

摘要: 由大型语言模型（LLM）驱动的自主代理的激增彻底改变了处理表格数据的流行业务应用程序，即片状药剂。尽管LLM被观察到容易受到来自外部数据源的即时注入攻击，但表格代理对攻击者的有效负载施加了严格的数据格式和预定义的规则，除非代理导航多层结构数据以合并有效负载，否则这些规则是无效的。为了应对这一挑战，我们提出了一种名为StruPhantom的新型攻击，该攻击专门针对黑匣子LLM供电的表格代理。我们的攻击设计了一个进化优化过程，该过程通过提出的由非主题评估器增强的约束蒙特卡洛树搜索来不断细化攻击有效负载。StruPhantom帮助系统性地探索和利用目标应用程序的弱点来实现目标劫持。我们的评估验证了StruPhantom在各种基于LLM的代理（包括现实世界平台上的代理）和攻击场景中的有效性。在强制应用程序响应以包含网络钓鱼链接或恶意代码方面，我们的攻击的成功率比基线高出50%以上。



## **22. An Investigation of Large Language Models and Their Vulnerabilities in Spam Detection**

垃圾邮件检测中的大型语言模型及其漏洞研究 cs.CR

10 pages; presented at HotSoS'2025 as a work in progress paper

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.09776v1) [paper-pdf](http://arxiv.org/pdf/2504.09776v1)

**Authors**: Qiyao Tang, Xiangyang Li

**Abstract**: Spam messages continue to present significant challenges to digital users, cluttering inboxes and posing security risks. Traditional spam detection methods, including rules-based, collaborative, and machine learning approaches, struggle to keep up with the rapidly evolving tactics employed by spammers. This project studies new spam detection systems that leverage Large Language Models (LLMs) fine-tuned with spam datasets. More importantly, we want to understand how LLM-based spam detection systems perform under adversarial attacks that purposefully modify spam emails and data poisoning attacks that exploit the differences between the training data and the massages in detection, to which traditional machine learning models are shown to be vulnerable. This experimentation employs two LLM models of GPT2 and BERT and three spam datasets of Enron, LingSpam, and SMSspamCollection for extensive training and testing tasks. The results show that, while they can function as effective spam filters, the LLM models are susceptible to the adversarial and data poisoning attacks. This research provides very useful insights for future applications of LLM models for information security.

摘要: 垃圾邮件继续给数字用户带来重大挑战，使收件箱变得杂乱并构成安全风险。传统的垃圾邮件检测方法，包括基于规则的、协作的和机器学习的方法，很难跟上垃圾邮件发送者所采用的快速发展的策略。该项目研究新的垃圾邮件检测系统，该系统利用经过垃圾邮件数据集微调的大型语言模型（LLM）。更重要的是，我们想了解基于LLM的垃圾邮件检测系统在有目的地修改垃圾邮件的对抗攻击和利用检测中训练数据和按摩之间差异的数据中毒攻击下如何表现，而传统的机器学习模型被证明是脆弱的。该实验使用两种LLM模型GPT 2和BERT以及三种垃圾邮件数据集Enron、LingSpam和SMSspamCollection来执行广泛的培训和测试任务。结果表明，虽然LLM模型可以充当有效的垃圾邮件过滤器，但它们很容易受到对抗性和数据中毒攻击。这项研究为LLM模型在信息安全方面的未来应用提供了非常有用的见解。



## **23. ControlNET: A Firewall for RAG-based LLM System**

Control NET：基于RAG的LLM系统的防火墙 cs.CR

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09593v1) [paper-pdf](http://arxiv.org/pdf/2504.09593v1)

**Authors**: Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang, Cong Wang, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.

摘要: 检索增强生成（RAG）显着增强了大型语言模型（LLM）的事实准确性和领域适应性。这一进步使它们能够在医疗保健、金融和企业应用程序等敏感领域广泛部署。RAG通过整合外部知识来缓解幻觉，但也会带来隐私风险和安全风险，尤其是数据泄露风险和数据中毒风险。虽然最近的研究探索了即时注射和中毒攻击，但在控制入站和出站查询流以减轻这些威胁的全面研究方面仍然存在显着差距。在本文中，我们提出了一种人工智能防火墙Controller NET，旨在保护基于RAG的LLM系统免受这些漏洞的影响。ControlNET通过利用激活转变现象来检测对抗性查询并通过语义分歧减轻其影响来控制查询流。我们使用最先进的开源LLM（Llama 3、Vicuna和Mistral）对四个不同的基准数据集（包括Mmarco、HotpotQA、FinQA和MedalSys）进行全面实验。我们的结果表明，ControlNET在检测和缓解安全威胁同时保持系统无害性方面达到了超过0.909 AUROC。总的来说，ControlNET提供了一种有效、健壮、无害的防御机制，标志着基于RAG的LLM系统安全部署的重大进步。



## **24. AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender**

AdaSteer：您的对齐LLM本质上是一个自适应越狱防御者 cs.CR

17 pages, 6 figures, 9 tables

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09466v1) [paper-pdf](http://arxiv.org/pdf/2504.09466v1)

**Authors**: Weixiang Zhao, Jiahe Guo, Yulin Hu, Yang Deng, An Zhang, Xingyu Sui, Xinyang Han, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu

**Abstract**: Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.

摘要: 尽管在安全调整方面做出了广泛的努力，但大型语言模型（LLM）仍然容易受到越狱攻击。激活转向提供了一种无需训练的防御方法，但依赖于固定的转向系数，从而导致次优保护和良性输入的错误拒绝增加。为了解决这个问题，我们提出了AdaSteer，这是一种自适应激活引导方法，可以根据输入特征动态调整模型行为。我们确定了两个关键属性：拒绝定律（R-Law），它表明与拒绝方向相反的越狱输入需要更强的引导，以及有害定律（H-Law），它区分对抗性和良性输入。AdaSteer沿着拒绝方向（RD）和有害方向（HD）引导输入表示，并通过逻辑回归学习自适应系数，确保强大的越狱防御，同时保留良性的输入处理。LLaMA-3.1、Gemma-2和Qwen 2.5的实验表明，AdaSteer在多次越狱攻击中优于基线方法，且对效用的影响最小。我们的结果强调了可解释模型内部要素在LLC中实时、灵活的安全执行方面的潜力。



## **25. SaRO: Enhancing LLM Safety through Reasoning-based Alignment**

SaRO：通过基于推理的一致提高LLM安全性 cs.CL

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09420v1) [paper-pdf](http://arxiv.org/pdf/2504.09420v1)

**Authors**: Yutao Mou, Yuxiao Luo, Shikun Zhang, Wei Ye

**Abstract**: Current safety alignment techniques for large language models (LLMs) face two key challenges: (1) under-generalization, which leaves models vulnerable to novel jailbreak attacks, and (2) over-alignment, which leads to the excessive refusal of benign instructions. Our preliminary investigation reveals semantic overlap between jailbreak/harmful queries and normal prompts in embedding space, suggesting that more effective safety alignment requires a deeper semantic understanding. This motivates us to incorporate safety-policy-driven reasoning into the alignment process. To this end, we propose the Safety-oriented Reasoning Optimization Framework (SaRO), which consists of two stages: (1) Reasoning-style Warmup (RW) that enables LLMs to internalize long-chain reasoning through supervised fine-tuning, and (2) Safety-oriented Reasoning Process Optimization (SRPO) that promotes safety reflection via direct preference optimization (DPO). Extensive experiments demonstrate the superiority of SaRO over traditional alignment methods.

摘要: 当前大型语言模型（LLM）的安全对齐技术面临两个关键挑战：（1）泛化不足，这使得模型容易受到新的越狱攻击，以及（2）过度对齐，这导致过度拒绝良性指令。我们的初步调查揭示了越狱/有害查询和嵌入空间中的正常提示之间的语义重叠，这表明更有效的安全对齐需要更深入的语义理解。这促使我们将安全策略驱动的推理纳入对齐过程。为此，我们提出了面向安全的推理优化框架（SaRO），它包括两个阶段：（1）推理式预热（RW），使LLM能够通过监督微调内化长链推理，以及（2）面向安全的推理过程优化（SRPO），通过直接偏好优化（DPO）促进安全反思。大量实验证明了SaRO相对于传统对齐方法的优越性。



## **26. Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities**

模型篡改攻击能够更严格地评估LLM能力 cs.CR

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2502.05209v2) [paper-pdf](http://arxiv.org/pdf/2502.05209v2)

**Authors**: Zora Che, Stephen Casper, Robert Kirk, Anirudh Satheesh, Stewart Slocum, Lev E McKinney, Rohit Gandikota, Aidan Ewart, Domenic Rosati, Zichu Wu, Zikui Cai, Bilal Chughtai, Yarin Gal, Furong Huang, Dylan Hadfield-Menell

**Abstract**: Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the attack success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.

摘要: 对大型语言模型（LLM）风险和能力的评估越来越多地被纳入人工智能风险管理和治理框架中。目前，大多数风险评估都是通过设计从系统中引发有害行为的输入来进行的。然而，这种方法有两个局限性。首先，投入产出评估无法评估开权模型的现实风险。其次，在任何特定的投入-产出评估期间识别的行为只能下限模型的最坏可能情况的投入-产出行为。作为引发有害行为的补充方法，我们建议使用模型篡改攻击来评估LLM，该攻击允许修改潜在激活或权重。我们使用最先进的技术来消除有害的LLM功能，以对抗一系列5个输入空间和6个模型篡改攻击。除了对这些方法进行比较之外，我们还表明：（1）模型对能力启发攻击的弹性取决于低维鲁棒性子空间;（2）模型篡改攻击的攻击成功率可以根据经验预测并为输入空间攻击的成功提供保守估计;和（3）最先进的取消学习方法可以在16个微调步骤内轻松取消。这些结果共同强调了抑制有害LLM功能的难度，并表明模型篡改攻击比单独的输入空间攻击可以实现更严格的评估。



## **27. Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities**

基于有限样本浓度不等式的LLM文本检测零次统计检验 stat.ML

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2501.02406v3) [paper-pdf](http://arxiv.org/pdf/2501.02406v3)

**Authors**: Tara Radvand, Mojtaba Abdolmaleki, Mohamed Mostagir, Ambuj Tewari

**Abstract**: Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly challenging as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. We answer the following question: Given a piece of text, can we identify whether it was produced by LLM $A$ or $B$ (where $B$ can be a human)? We model LLM-generated text as a sequential stochastic process with complete dependence on history and design zero-shot statistical tests to distinguish between (i) the text generated by two different sets of LLMs $A$ (in-house) and $B$ (non-sanctioned) and also (ii) LLM-generated and human-generated texts. We prove that our tests' type I and type II errors decrease exponentially as text length increases. For designing our tests for a given string, we demonstrate that if the string is generated by the evaluator model $A$, the log-perplexity of the string under $A$ converges to the average entropy of the string under $A$, except with an exponentially small probability in the string length. We also show that if $B$ generates the text, except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. For our experiments: First, we present experiments using open-source LLMs to support our theoretical results, and then we provide experiments in a black-box setting with adversarial attacks. Practically, our work enables guaranteed finding of the origin of harmful or false LLM-generated text, which can be useful for combating misinformation and compliance with emerging AI regulations.

摘要: 验证内容的出处对于许多组织的功能至关重要，例如，教育机构、社交媒体平台、公司等。随着大型语言模型（LLM）生成的文本与人类生成的内容几乎无法区分，这个问题变得越来越具有挑战性。此外，许多机构利用内部LLM，并希望确保外部未经批准的LLM不会在机构内制作内容。我们回答以下问题：给定一段文本，我们可以识别它是由LLM $A$还是$B$（其中$B$可以是人类）生成的吗？我们将LLM生成的文本建模为一个顺序随机过程，完全依赖于历史，并设计零次统计测试来区分（i）由两组不同的LLM $A$（内部）和$B$（非认可）生成的文本，以及（ii）LLM生成的文本和人类生成的文本。我们证明了我们的测试的类型I和类型II错误随着文本长度的增加而呈指数级下降。为了设计我们的测试对于一个给定的字符串，我们证明，如果字符串是由评估模型$A$，下$A$的字符串的对数困惑收敛到下$A$的字符串的平均熵，除了在字符串长度的指数小的概率。我们还表明，如果$B$生成的文本，除了一个指数小概率的字符串长度，下$A$的字符串的对数困惑收敛到平均交叉熵的$B$和$A$。对于我们的实验：首先，我们使用开源LLM进行实验来支持我们的理论结果，然后我们在具有对抗性攻击的黑匣子环境中提供实验。实际上，我们的工作能够保证找到LLM生成的有害或虚假文本的来源，这对于打击错误信息和遵守新出现的人工智能法规非常有用。



## **28. Feature-Aware Malicious Output Detection and Mitigation**

具有攻击意识的恶意输出检测和缓解 cs.CL

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2504.09191v1) [paper-pdf](http://arxiv.org/pdf/2504.09191v1)

**Authors**: Weilong Dong, Peiguang Li, Yu Tian, Xinyi Zeng, Fengdi Li, Sirui Wang

**Abstract**: The rapid advancement of large language models (LLMs) has brought significant benefits to various domains while introducing substantial risks. Despite being fine-tuned through reinforcement learning, LLMs lack the capability to discern malicious content, limiting their defense against jailbreak. To address these safety concerns, we propose a feature-aware method for harmful response rejection (FMM), which detects the presence of malicious features within the model's feature space and adaptively adjusts the model's rejection mechanism. By employing a simple discriminator, we detect potential malicious traits during the decoding phase. Upon detecting features indicative of toxic tokens, FMM regenerates the current token. By employing activation patching, an additional rejection vector is incorporated during the subsequent token generation, steering the model towards a refusal response. Experimental results demonstrate the effectiveness of our approach across multiple language models and diverse attack techniques, while crucially maintaining the models' standard generation capabilities.

摘要: 大型语言模型（LLM）的快速发展为各个领域带来了巨大的好处，同时也带来了巨大的风险。尽管通过强化学习进行了微调，但LLM缺乏识别恶意内容的能力，限制了他们对越狱的防御。为了解决这些安全问题，我们提出了一种用于有害响应拒绝（FMM）的特征感知方法，该方法检测模型特征空间内是否存在恶意特征，并自适应地调整模型的拒绝机制。通过使用简单的收件箱，我们在解码阶段检测潜在的恶意特征。检测到指示有毒令牌的特征后，FMM会重新生成当前令牌。通过使用激活补丁，在后续令牌生成期间合并额外的拒绝载体，引导模型转向拒绝响应。实验结果证明了我们的方法在多种语言模型和多种攻击技术中的有效性，同时至关重要地保持了模型的标准生成能力。



## **29. Privacy Preservation in Gen AI Applications**

世代人工智能应用程序中的隐私保护 cs.CR

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2504.09095v1) [paper-pdf](http://arxiv.org/pdf/2504.09095v1)

**Authors**: Swetha S, Ram Sundhar K Shaju, Rakshana M, Ganesh R, Balavedhaa S, Thiruvaazhi U

**Abstract**: The ability of machines to comprehend and produce language that is similar to that of humans has revolutionized sectors like customer service, healthcare, and finance thanks to the quick advances in Natural Language Processing (NLP), which are fueled by Generative Artificial Intelligence (AI) and Large Language Models (LLMs). However, because LLMs trained on large datasets may unintentionally absorb and reveal Personally Identifiable Information (PII) from user interactions, these capabilities also raise serious privacy concerns. Deep neural networks' intricacy makes it difficult to track down or stop the inadvertent storing and release of private information, which raises serious concerns about the privacy and security of AI-driven data. This study tackles these issues by detecting Generative AI weaknesses through attacks such as data extraction, model inversion, and membership inference. A privacy-preserving Generative AI application that is resistant to these assaults is then developed. It ensures privacy without sacrificing functionality by using methods to identify, alter, or remove PII before to dealing with LLMs. In order to determine how well cloud platforms like Microsoft Azure, Google Cloud, and AWS provide privacy tools for protecting AI applications, the study also examines these technologies. In the end, this study offers a fundamental privacy paradigm for generative AI systems, focusing on data security and moral AI implementation, and opening the door to a more secure and conscientious use of these tools.

摘要: 由于自然语言处理（NLP）的快速发展，机器理解和产生与人类类似的语言的能力彻底改变了客户服务、医疗保健和金融等领域，这要归功于生成人工智能（AI）和大型语言模型（LLM）。然而，由于在大型数据集上训练的LLM可能会无意中吸收和泄露来自用户交互的个人可识别信息（PRI），因此这些功能也会引发严重的隐私问题。深度神经网络的复杂性使得很难追踪或阻止私人信息的无意存储和发布，这引发了人们对人工智能驱动数据的隐私和安全性的严重担忧。这项研究通过数据提取、模型倒置和隶属度推断等攻击来检测生成性人工智能的弱点来解决这些问题。然后开发出一个能够抵抗这些攻击的保护隐私的生成人工智能应用程序。它通过在处理LLM之前使用识别、更改或删除PRI的方法来确保隐私，而不会牺牲功能。为了确定Microsoft Azure、Google Cloud和AWS等云平台为保护人工智能应用程序提供的隐私工具的效果如何，该研究还研究了这些技术。最后，这项研究为生成性人工智能系统提供了一个基本的隐私范式，重点关注数据安全和道德人工智能实施，并为更安全、更认真地使用这些工具打开了大门。



## **30. Detecting Instruction Fine-tuning Attack on Language Models with Influence Function**

利用影响函数检测语言模型的指令微调攻击 cs.LG

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2504.09026v1) [paper-pdf](http://arxiv.org/pdf/2504.09026v1)

**Authors**: Jiawei Li

**Abstract**: Instruction fine-tuning attacks pose a significant threat to large language models (LLMs) by subtly embedding poisoned data in fine-tuning datasets, which can trigger harmful or unintended responses across a range of tasks. This undermines model alignment and poses security risks in real-world deployment. In this work, we present a simple and effective approach to detect and mitigate such attacks using influence functions, a classical statistical tool adapted for machine learning interpretation. Traditionally, the high computational costs of influence functions have limited their application to large models and datasets. The recent Eigenvalue-Corrected Kronecker-Factored Approximate Curvature (EK-FAC) approximation method enables efficient influence score computation, making it feasible for large-scale analysis.   We are the first to apply influence functions for detecting language model instruction fine-tuning attacks on large-scale datasets, as both the instruction fine-tuning attack on language models and the influence calculation approximation technique are relatively new. Our large-scale empirical evaluation of influence functions on 50,000 fine-tuning examples and 32 tasks reveals a strong association between influence scores and sentiment. Building on this, we introduce a novel sentiment transformation combined with influence functions to detect and remove critical poisons -- poisoned data points that skew model predictions. Removing these poisons (only 1% of total data) recovers model performance to near-clean levels, demonstrating the effectiveness and efficiency of our approach. Artifact is available at https://github.com/lijiawei20161002/Poison-Detection.   WARNING: This paper contains offensive data examples.

摘要: 指令微调攻击通过巧妙地将有毒数据嵌入微调数据集中，对大型语言模型（LLM）构成重大威胁，这可能会在一系列任务中触发有害或意外的响应。这会破坏模型一致性，并在现实世界部署中带来安全风险。在这项工作中，我们提出了一种简单有效的方法来使用影响函数来检测和减轻此类攻击，影响函数是一种适合机器学习解释的经典统计工具。传统上，影响函数的高计算成本限制了其对大型模型和数据集的应用。最近的特征值修正克罗内克因子逼近曲线（EK-FAC）逼近方法能够实现高效的影响分数计算，使其适合大规模分析。   我们是第一个应用影响函数来检测对大规模数据集的语言模型指令微调攻击的人，因为对语言模型的指令微调攻击和影响计算逼近技术都是相对较新的。我们对50，000个微调示例和32个任务的影响力函数进行了大规模实证评估，揭示了影响力分数和情绪之间存在很强的关联。在此基础上，我们引入了一种新颖的情感转换，并结合影响函数来检测和删除关键毒药--扭曲模型预测的有毒数据点。删除这些毒物（仅占总数据的1%）可将模型性能恢复到接近清洁的水平，证明了我们方法的有效性和效率。收件箱可在https://github.com/lijiawei20161002/Poison-Detection上获取。   警告：本文包含攻击性数据示例。



## **31. Robust Steganography from Large Language Models**

来自大型语言模型的稳健隐写术 cs.CR

36 pages, 9 figures

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08977v1) [paper-pdf](http://arxiv.org/pdf/2504.08977v1)

**Authors**: Neil Perry, Sanket Gupte, Nishant Pitta, Lior Rotem

**Abstract**: Recent steganographic schemes, starting with Meteor (CCS'21), rely on leveraging large language models (LLMs) to resolve a historically-challenging task of disguising covert communication as ``innocent-looking'' natural-language communication. However, existing methods are vulnerable to ``re-randomization attacks,'' where slight changes to the communicated text, that might go unnoticed, completely destroy any hidden message. This is also a vulnerability in more traditional encryption-based stegosystems, where adversaries can modify the randomness of an encryption scheme to destroy the hidden message while preserving an acceptable covertext to ordinary users. In this work, we study the problem of robust steganography. We introduce formal definitions of weak and strong robust LLM-based steganography, corresponding to two threat models in which natural language serves as a covertext channel resistant to realistic re-randomization attacks. We then propose two constructions satisfying these notions. We design and implement our steganographic schemes that embed arbitrary secret messages into natural language text generated by LLMs, ensuring recoverability even under adversarial paraphrasing and rewording attacks. To support further research and real-world deployment, we release our implementation and datasets for public use.

摘要: 最近的隐写计划，从Meteor（CCS ' 21）开始，依靠利用大型语言模型（LLM）来解决一项具有历史挑战性的任务，即将秘密通信伪装成“看起来无辜”的自然语言通信。然而，现有的方法很容易受到“重新随机化攻击”，即对所传达的文本的轻微变化（可能不被注意到）会完全破坏任何隐藏的信息。这也是更传统的基于加密的隐写系统中的一个漏洞，其中对手可以修改加密方案的随机性以破坏隐藏消息，同时为普通用户保留可接受的封面文本。在这项工作中，我们研究了稳健隐写术的问题。我们介绍了正式定义的弱和强鲁棒的基于LLM的隐写术，对应于两个威胁模型，其中自然语言作为一个covertext信道抵抗现实的重新随机化攻击。然后，我们提出了两个建设满足这些概念。我们设计并实现了隐写方案，将任意秘密消息嵌入到LLM生成的自然语言文本中，即使在对抗性释义和改写攻击下也能确保可恢复性。为了支持进一步的研究和实际部署，我们发布了我们的实现和数据集供公众使用。



## **32. Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups**

穿越兔子洞：LLM生成的针对心理健康群体的攻击叙事中的紧急偏见 cs.CL

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.06160v3) [paper-pdf](http://arxiv.org/pdf/2504.06160v3)

**Authors**: Rijul Magu, Arka Dutta, Sean Kim, Ashiqur R. KhudaBukhsh, Munmun De Choudhury

**Abstract**: Large Language Models (LLMs) have been shown to demonstrate imbalanced biases against certain groups. However, the study of unprovoked targeted attacks by LLMs towards at-risk populations remains underexplored. Our paper presents three novel contributions: (1) the explicit evaluation of LLM-generated attacks on highly vulnerable mental health groups; (2) a network-based framework to study the propagation of relative biases; and (3) an assessment of the relative degree of stigmatization that emerges from these attacks. Our analysis of a recently released large-scale bias audit dataset reveals that mental health entities occupy central positions within attack narrative networks, as revealed by a significantly higher mean centrality of closeness (p-value = 4.06e-10) and dense clustering (Gini coefficient = 0.7). Drawing from sociological foundations of stigmatization theory, our stigmatization analysis indicates increased labeling components for mental health disorder-related targets relative to initial targets in generation chains. Taken together, these insights shed light on the structural predilections of large language models to heighten harmful discourse and highlight the need for suitable approaches for mitigation.

摘要: 事实证明，大型语言模型（LLM）对某些群体表现出不平衡的偏见。然而，关于LLM对高危人群进行无端有针对性攻击的研究仍然没有得到充分的研究。我们的论文提出了三个新颖的贡献：（1）对LLM产生的对高度弱势心理健康群体的攻击的明确评估;（2）基于网络的框架来研究相对偏见的传播;（3）对这些攻击中出现的耻辱的相对程度的评估。我们对最近发布的大规模偏见审计数据集的分析表明，心理健康实体在攻击叙事网络中占据了中心位置，这一点表现为密切度（p值= 4.06e-10）和密集聚集度（基尼系数= 0.7）的平均中心性显着更高。根据污名化理论的社会学基础，我们的污名化分析表明，相对于代际链中的初始目标，与心理健康疾病相关目标的标签成分有所增加。总而言之，这些见解揭示了大型语言模型加剧有害话语的结构偏好，并强调了适当的缓解方法的必要性。



## **33. MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits**

HCP安全审计：具有模型上下文协议的LLM允许重大安全漏洞 cs.CR

27 pages, 21 figures, and 2 Tables. Cleans up the TeX source

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.03767v2) [paper-pdf](http://arxiv.org/pdf/2504.03767v2)

**Authors**: Brandon Radosevich, John Halloran

**Abstract**: To reduce development overhead and enable seamless integration between potential components comprising any given generative AI application, the Model Context Protocol (MCP) (Anthropic, 2024) has recently been released and subsequently widely adopted. The MCP is an open protocol that standardizes API calls to large language models (LLMs), data sources, and agentic tools. By connecting multiple MCP servers, each defined with a set of tools, resources, and prompts, users are able to define automated workflows fully driven by LLMs. However, we show that the current MCP design carries a wide range of security risks for end users. In particular, we demonstrate that industry-leading LLMs may be coerced into using MCP tools to compromise an AI developer's system through various attacks, such as malicious code execution, remote access control, and credential theft. To proactively mitigate these and related attacks, we introduce a safety auditing tool, MCPSafetyScanner, the first agentic tool to assess the security of an arbitrary MCP server. MCPScanner uses several agents to (a) automatically determine adversarial samples given an MCP server's tools and resources; (b) search for related vulnerabilities and remediations based on those samples; and (c) generate a security report detailing all findings. Our work highlights serious security issues with general-purpose agentic workflows while also providing a proactive tool to audit MCP server safety and address detected vulnerabilities before deployment.   The described MCP server auditing tool, MCPSafetyScanner, is freely available at: https://github.com/johnhalloran321/mcpSafetyScanner

摘要: 为了减少开发费用并实现构成任何给定生成式人工智能应用程序的潜在组件之间的无缝集成，模型上下文协议（HCP）（Anthropic，2024）最近发布并随后广泛采用。HCP是一种开放协议，可同步化对大型语言模型（LLM）、数据源和代理工具的API调用。通过连接多个HCP服务器（每个服务器都定义了一组工具、资源和提示），用户能够定义完全由LLM驱动的自动化工作流程。然而，我们表明当前的LCP设计对最终用户来说存在广泛的安全风险。特别是，我们证明了行业领先的LLM可能会被迫使用LCP工具通过各种攻击（例如恶意代码执行、远程访问控制和凭证盗窃）来危害人工智能开发人员的系统。为了主动缓解这些攻击和相关攻击，我们引入了安全审计工具MCPSafetyScanner，这是第一个评估任意LCP服务器安全性的代理工具。MCPScanner使用多个代理来（a）在给定HCP服务器的工具和资源的情况下自动确定对抗样本;（b）根据这些样本搜索相关漏洞和补救措施;以及（c）生成详细说明所有发现结果的安全报告。我们的工作强调了通用代理工作流程的严重安全问题，同时还提供了一种主动工具来审计LCP服务器的安全性并在部署之前解决检测到的漏洞。   所描述的LCP服务器审计工具MCPSafetyScanner可在以下网址免费获取：https://github.com/johnhalloran321/mcpSafetyScanner



## **34. SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models**

SCAM：多模式基础模型的真实印刷稳健性评估 cs.CV

Submitted to CVPR 2025 Workshop EVAL-FoMo-2

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.04893v2) [paper-pdf](http://arxiv.org/pdf/2504.04893v2)

**Authors**: Justus Westerhoff, Erblina Purelku, Jakob Hackstein, Leo Pinetzki, Lorenz Hufe

**Abstract**: Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper under https://huggingface.co/datasets/BLISS-e-V/SCAM, along with the code for evaluations at https://github.com/Bliss-e-V/SCAM.

摘要: 印刷攻击利用多模式基础模型中文本和视觉内容之间的相互作用，当误导性文本嵌入图像中时，会导致错误分类。然而，现有数据集的大小和多样性有限，因此很难研究此类漏洞。在本文中，我们介绍了SCAM，这是迄今为止最大、最多样化的现实世界印刷攻击图像数据集，包含涵盖数百个对象类别和攻击词的1，162张图像。通过对SCAM上的视觉语言模型（VLM）进行广泛的基准测试，我们证明了印刷攻击会显着降低性能，并确定训练数据和模型架构会影响对这些攻击的易感性。我们的研究结果表明，由于视觉编码器的选择，印刷攻击在最先进的大型视觉语言模型（LVLM）中持续存在，尽管更大的大型语言模型（LLM）主干有助于减轻它们的脆弱性。此外，我们还证明合成攻击与现实世界（手写）攻击非常相似，验证了它们在研究中的用途。我们的工作提供了全面的资源和经验见解，以促进未来对稳健且值得信赖的多模式人工智能系统的研究。我们在https：//huggingface.co/guardets/BLISS-e-V/SCAM下公开发布本文中介绍的数据集，以及https://github.com/Bliss-e-V/SCAM上的评估代码。



## **35. X-Guard: Multilingual Guard Agent for Content Moderation**

X-Guard：用于内容审核的多语言Guard代理 cs.CR

34 pages, 15 figures

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08848v1) [paper-pdf](http://arxiv.org/pdf/2504.08848v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan, Ph. D

**Abstract**: Large Language Models (LLMs) have rapidly become integral to numerous applications in critical domains where reliability is paramount. Despite significant advances in safety frameworks and guardrails, current protective measures exhibit crucial vulnerabilities, particularly in multilingual contexts. Existing safety systems remain susceptible to adversarial attacks in low-resource languages and through code-switching techniques, primarily due to their English-centric design. Furthermore, the development of effective multilingual guardrails is constrained by the scarcity of diverse cross-lingual training data. Even recent solutions like Llama Guard-3, while offering multilingual support, lack transparency in their decision-making processes. We address these challenges by introducing X-Guard agent, a transparent multilingual safety agent designed to provide content moderation across diverse linguistic contexts. X-Guard effectively defends against both conventional low-resource language attacks and sophisticated code-switching attacks. Our approach includes: curating and enhancing multiple open-source safety datasets with explicit evaluation rationales; employing a jury of judges methodology to mitigate individual judge LLM provider biases; creating a comprehensive multilingual safety dataset spanning 132 languages with 5 million data points; and developing a two-stage architecture combining a custom-finetuned mBART-50 translation module with an evaluation X-Guard 3B model trained through supervised finetuning and GRPO training. Our empirical evaluations demonstrate X-Guard's effectiveness in detecting unsafe content across multiple languages while maintaining transparency throughout the safety evaluation process. Our work represents a significant advancement in creating robust, transparent, and linguistically inclusive safety systems for LLMs and its integrated systems.

摘要: 大型语言模型（LLM）已迅速成为可靠性至关重要的关键领域众多应用程序的组成部分。尽管安全框架和护栏取得了重大进展，但当前的保护措施仍表现出严重的漏洞，特别是在多语言环境中。现有的安全系统仍然容易受到低资源语言和代码切换技术的对抗攻击，这主要是由于其以英语为中心的设计。此外，有效的多语言护栏的开发受到多样化跨语言培训数据的稀缺的限制。即使是Llama Guard-3等最近的解决方案，虽然提供了多语言支持，但其决策过程缺乏透明度。我们通过引入X-Guard代理来应对这些挑战，X-Guard代理是一种透明的多语言安全代理，旨在提供跨不同语言上下文的内容审核。X-Guard可以有效防御传统的低资源语言攻击和复杂的代码交换攻击。我们的方法包括：利用明确的评估原理来策划和增强多个开源安全数据集;采用法官陪审团方法来减轻个别法官LLM提供商的偏见;创建涵盖132种语言、包含500万个数据点的全面多语言安全数据集;并开发将定制微调mBART-50翻译模块与评估X-相结合的两阶段架构Guard 3B模型通过监督微调和GRPO培训进行培训。我们的实证评估证明了X-Guard在检测多种语言的不安全内容方面的有效性，同时在整个安全评估过程中保持透明度。我们的工作代表着为LLM及其集成系统创建强大、透明且语言包容的安全系统方面的重大进步。



## **36. GenXSS: an AI-Driven Framework for Automated Detection of XSS Attacks in WAFs**

GenXSS：用于自动检测WAF中XSS攻击的人工智能驱动框架 cs.CR

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08176v1) [paper-pdf](http://arxiv.org/pdf/2504.08176v1)

**Authors**: Vahid Babaey, Arun Ravindran

**Abstract**: The increasing reliance on web services has led to a rise in cybersecurity threats, particularly Cross-Site Scripting (XSS) attacks, which target client-side layers of web applications by injecting malicious scripts. Traditional Web Application Firewalls (WAFs) struggle to detect highly obfuscated and complex attacks, as their rules require manual updates. This paper presents a novel generative AI framework that leverages Large Language Models (LLMs) to enhance XSS mitigation. The framework achieves two primary objectives: (1) generating sophisticated and syntactically validated XSS payloads using in-context learning, and (2) automating defense mechanisms by testing these attacks against a vulnerable application secured by a WAF, classifying bypassing attacks, and generating effective WAF security rules. Experimental results using GPT-4o demonstrate the framework's effectiveness generating 264 XSS payloads, 83% of which were validated, with 80% bypassing ModSecurity WAF equipped with an industry standard security rule set developed by the Open Web Application Security Project (OWASP) to protect against web vulnerabilities. Through rule generation, 86% of previously successful attacks were blocked using only 15 new rules. In comparison, Google Gemini Pro achieved a lower bypass rate of 63%, highlighting performance differences across LLMs.

摘要: 对网络服务的依赖日益增加，导致网络安全威胁的增加，特别是跨站点脚本（XSS）攻击，这些攻击通过注入恶意脚本来针对网络应用程序的客户端层。传统的Web应用程序防火墙（WAF）很难检测高度模糊和复杂的攻击，因为它们的规则需要手动更新。本文提出了一种新颖的生成式人工智能框架，该框架利用大型语言模型（LLM）来增强XSS缓解。该框架实现了两个主要目标：（1）使用上下文学习生成复杂且经过语法验证的XSS有效负载，以及（2）通过针对WAF保护的脆弱应用程序测试这些攻击、对绕过攻击进行分类并生成有效的WAF安全规则来自动化防御机制。使用GPT-4 o的实验结果证明了该框架的有效性，生成264个XSS有效负载，其中83%经过验证，80%绕过ModSecurity WAF，该WAF配备了开放Web应用程序安全项目（OWISP）开发的行业标准安全规则集，以防止网络漏洞。通过规则生成，仅使用15个新规则就阻止了86%之前成功的攻击。相比之下，Google Gemini Pro的绕过率较低，为63%，凸显了LLM之间的性能差异。



## **37. Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge**

大型语言模型中对偏见激发的对抗鲁棒性进行基准测试：利用LLM作为评委的可扩展自动化评估 cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07887v1) [paper-pdf](http://arxiv.org/pdf/2504.07887v1)

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence, driving advancements in machine translation, summarization, and conversational agents. However, their increasing integration into critical societal domains has raised concerns about embedded biases, which can perpetuate stereotypes and compromise fairness. These biases stem from various sources, including historical inequalities in training data, linguistic imbalances, and adversarial manipulation. Despite mitigation efforts, recent studies indicate that LLMs remain vulnerable to adversarial attacks designed to elicit biased responses. This work proposes a scalable benchmarking framework to evaluate LLM robustness against adversarial bias elicitation. Our methodology involves (i) systematically probing models with a multi-task approach targeting biases across various sociocultural dimensions, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach for automated assessment of model responses, and (iii) employing jailbreak techniques to investigate vulnerabilities in safety mechanisms. Our analysis examines prevalent biases in both small and large state-of-the-art models and their impact on model safety. Additionally, we assess the safety of domain-specific models fine-tuned for critical fields, such as medicine. Finally, we release a curated dataset of bias-related prompts, CLEAR-Bias, to facilitate systematic vulnerability benchmarking. Our findings reveal critical trade-offs between model size and safety, aiding the development of fairer and more robust future language models.

摘要: 大型语言模型（LLM）彻底改变了人工智能，推动了机器翻译、摘要和对话代理的进步。然而，他们越来越融入关键社会领域，引发了人们对根深蒂固的偏见的担忧，这可能会延续刻板印象并损害公平性。这些偏见源于各种来源，包括训练数据的历史不平等、语言不平衡和对抗操纵。尽管做出了缓解措施，但最近的研究表明，LLM仍然容易受到旨在引发偏见反应的对抗攻击。这项工作提出了一个可扩展的基准测试框架，以评估LLM针对对抗性偏见引发的稳健性。我们的方法包括：（i）采用针对各个社会文化维度的偏见的多任务方法系统地探索模型，（ii）使用LLM作为法官的方法通过安全评分量化稳健性，以自动评估模型响应，以及（iii）采用越狱技术来调查安全机制中的漏洞。我们的分析检查了小型和大型最先进模型中普遍存在的偏差及其对模型安全性的影响。此外，我们还评估针对医学等关键领域进行微调的特定领域模型的安全性。最后，我们发布了一个精心策划的偏差相关提示数据集ClearAR-Bias，以促进系统性漏洞基准测试。我们的研究结果揭示了模型大小和安全性之间的关键权衡，有助于开发更公平、更强大的未来语言模型。



## **38. PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization**

PR攻击：通过二层优化对大型语言模型中的检索增强生成进行协调的预算-RAG攻击 cs.CR

Accepted at SIGIR 2025

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07717v1) [paper-pdf](http://arxiv.org/pdf/2504.07717v1)

**Authors**: Yang Jiao, Xiaodong Wang, Kai Yang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of applications, e.g., medical question-answering, mathematical sciences, and code generation. However, they also exhibit inherent limitations, such as outdated knowledge and susceptibility to hallucinations. Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to address these issues, but it also introduces new vulnerabilities. Recent efforts have focused on the security of RAG-based LLMs, yet existing attack methods face three critical challenges: (1) their effectiveness declines sharply when only a limited number of poisoned texts can be injected into the knowledge database, (2) they lack sufficient stealth, as the attacks are often detectable by anomaly detection systems, which compromises their effectiveness, and (3) they rely on heuristic approaches to generate poisoned texts, lacking formal optimization frameworks and theoretic guarantees, which limits their effectiveness and applicability. To address these issues, we propose coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack that introduces a small number of poisoned texts into the knowledge database while embedding a backdoor trigger within the prompt. When activated, the trigger causes the LLM to generate pre-designed responses to targeted queries, while maintaining normal behavior in other contexts. This ensures both high effectiveness and stealth. We formulate the attack generation process as a bilevel optimization problem leveraging a principled optimization framework to develop optimal poisoned texts and triggers. Extensive experiments across diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving a high attack success rate even with a limited number of poisoned texts and significantly improved stealth compared to existing methods.

摘要: 大型语言模型（LLM）已在广泛的应用程序中表现出出色的性能，例如，医学问答、数学科学和代码生成。然而，它们也表现出固有的局限性，例如过时的知识和幻觉的易感性。检索增强一代（RAG）已成为解决这些问题的一个有希望的范式，但它也引入了新的漏洞。最近的工作重点是基于RAG的LLM的安全性，但现有的攻击方法面临三个关键挑战：（1）当只能将有限数量的有毒文本注入知识数据库时，它们的有效性急剧下降，（2）它们缺乏足够的隐蔽性，因为异常检测系统通常可以检测到攻击，这损害了它们的有效性，（3）它们依赖启发式方法来生成有毒文本，缺乏正式的优化框架和理论保证，这限制了它们的有效性和适用性。为了解决这些问题，我们提出了协调的预算-RAG攻击（PR-攻击），这是一种新型的优化驱动攻击，它将少量有毒文本引入知识数据库，同时在提示内嵌入后门触发器。激活时，触发器会使LLM生成对目标查询的预先设计的响应，同时在其他上下文中保持正常行为。这确保了高效率和隐形性。我们将攻击生成过程制定为一个双层优化问题，利用有原则的优化框架来开发最佳的有毒文本和触发器。跨不同LLM和数据集的广泛实验证明了PR-Attack的有效性，即使在数量有限的有毒文本的情况下也能实现很高的攻击成功率，并且与现有方法相比，隐身性显着提高。



## **39. Agent That Debugs: Dynamic State-Guided Vulnerability Repair**

调试代理：动态状态引导的漏洞修复 cs.SE

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07634v1) [paper-pdf](http://arxiv.org/pdf/2504.07634v1)

**Authors**: Zhengyao Liu, Yunlong Ma, Jingxuan Xu, Junchen Ai, Xiang Gao, Hailong Sun, Abhik Roychoudhury

**Abstract**: In recent years, more vulnerabilities have been discovered every day, while manual vulnerability repair requires specialized knowledge and is time-consuming. As a result, many detected or even published vulnerabilities remain unpatched, thereby increasing the exposure of software systems to attacks. Recent advancements in agents based on Large Language Models have demonstrated their increasing capabilities in code understanding and generation, which can be promising to achieve automated vulnerability repair. However, the effectiveness of agents based on static information retrieval is still not sufficient for patch generation. To address the challenge, we propose a program repair agent called VulDebugger that fully utilizes both static and dynamic context, and it debugs programs in a manner akin to humans. The agent inspects the actual state of the program via the debugger and infers expected states via constraints that need to be satisfied. By continuously comparing the actual state with the expected state, it deeply understands the root causes of the vulnerabilities and ultimately accomplishes repairs. We experimentally evaluated VulDebugger on 50 real-life projects. With 60.00% successfully fixed, VulDebugger significantly outperforms state-of-the-art approaches for vulnerability repair.

摘要: 近年来，每天都有更多的漏洞被发现，而人工漏洞修复需要专业知识，耗时长。因此，许多检测到的甚至公布的漏洞仍然没有修补，从而增加了软件系统遭受攻击的风险。基于大型语言模型的代理的最新进展已经证明了它们在代码理解和生成方面的能力越来越强，这可能有望实现自动漏洞修复。然而，基于静态信息检索的代理的有效性仍然是不够的补丁生成。为了应对这一挑战，我们提出了一种名为Vulligger的程序修复代理，它充分利用静态和动态上下文，并以类似于人类的方式调试程序。代理通过调试器检查程序的实际状态，并通过需要满足的约束推断预期状态。通过不断比较实际状态与预期状态，深刻了解漏洞的根本原因并最终完成修复。我们对50个现实生活中的项目进行了实验性评估。Vulligger成功修复60.00%，其性能明显优于最先进的漏洞修复方法。



## **40. Defense against Prompt Injection Attacks via Mixture of Encodings**

通过混合编码防御即时注入攻击 cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07467v1) [paper-pdf](http://arxiv.org/pdf/2504.07467v1)

**Authors**: Ruiyi Zhang, David Sullivan, Kyle Jackson, Pengtao Xie, Mei Chen

**Abstract**: Large Language Models (LLMs) have emerged as a dominant approach for a wide range of NLP tasks, with their access to external information further enhancing their capabilities. However, this introduces new vulnerabilities, known as prompt injection attacks, where external content embeds malicious instructions that manipulate the LLM's output. Recently, the Base64 defense has been recognized as one of the most effective methods for reducing success rate of prompt injection attacks. Despite its efficacy, this method can degrade LLM performance on certain NLP tasks. To address this challenge, we propose a novel defense mechanism: mixture of encodings, which utilizes multiple character encodings, including Base64. Extensive experimental results show that our method achieves one of the lowest attack success rates under prompt injection attacks, while maintaining high performance across all NLP tasks, outperforming existing character encoding-based defense methods. This underscores the effectiveness of our mixture of encodings strategy for both safety and task performance metrics.

摘要: 大型语言模型（LLM）已成为各种NLP任务的主要方法，它们对外部信息的访问进一步增强了它们的能力。然而，这会引入新的漏洞，称为提示注入攻击，其中外部内容嵌入了操纵LLM输出的恶意指令。最近，Base 64防御被公认为降低即时注入攻击成功率的最有效方法之一。尽管该方法有效，但它可能会降低LLM在某些NLP任务上的性能。为了应对这一挑战，我们提出了一种新颖的防御机制：混合编码，它利用多种字符编码，包括Base 64。大量实验结果表明，我们的方法在提示注入攻击下实现了攻击成功率最低的方法之一，同时在所有NLP任务中保持高性能，优于现有的基于字符编码的防御方法。这凸显了我们混合编码策略对于安全和任务性能指标的有效性。



## **41. Achilles Heel of Distributed Multi-Agent Systems**

分布式多智能体系统的致命弱点 cs.MA

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07461v1) [paper-pdf](http://arxiv.org/pdf/2504.07461v1)

**Authors**: Yiting Zhang, Yijiang Li, Tianwei Zhao, Kaijie Zhu, Haohan Wang, Nuno Vasconcelos

**Abstract**: Multi-agent system (MAS) has demonstrated exceptional capabilities in addressing complex challenges, largely due to the integration of multiple large language models (LLMs). However, the heterogeneity of LLMs, the scalability of quantities of LLMs, and local computational constraints pose significant challenges to hosting these models locally. To address these issues, we propose a new framework termed Distributed Multi-Agent System (DMAS). In DMAS, heterogeneous third-party agents function as service providers managed remotely by a central MAS server and each agent offers its services through API interfaces. However, the distributed nature of DMAS introduces several concerns about trustworthiness. In this paper, we study the Achilles heel of distributed multi-agent systems, identifying four critical trustworthiness challenges: free riding, susceptibility to malicious attacks, communication inefficiencies, and system instability. Extensive experiments across seven frameworks and four datasets reveal significant vulnerabilities of the DMAS. These attack strategies can lead to a performance degradation of up to 80% and attain a 100% success rate in executing free riding and malicious attacks. We envision our work will serve as a useful red-teaming tool for evaluating future multi-agent systems and spark further research on trustworthiness challenges in distributed multi-agent systems.

摘要: 多代理系统（MAS）在应对复杂挑战方面表现出了卓越的能力，这主要归功于多个大型语言模型（LLM）的集成。然而，LLM的多样性、LLM数量的可扩展性以及本地计算限制对本地托管这些模型构成了重大挑战。为了解决这些问题，我们提出了一个名为分布式多代理系统（DMAS）的新框架。在DMAS中，异类第三方代理充当由中央MAS服务器远程管理的服务提供商，每个代理通过API接口提供服务。然而，DMAS的分布式特性引入了一些关于可信度的问题。在本文中，我们研究了分布式多智能体系统的致命弱点，确定了四个关键的可信性挑战：搭便车，易受恶意攻击，通信效率低下，系统不稳定。在七个框架和四个数据集上进行的广泛实验揭示了DMAS的重大漏洞。这些攻击策略可以导致高达80%的性能下降，并在执行搭便车和恶意攻击时达到100%的成功率。我们设想我们的工作将成为评估未来多代理系统的有用的红色团队工具，并引发对分布式多代理系统可信度挑战的进一步研究。



## **42. Code Generation with Small Language Models: A Deep Evaluation on Codeforces**

使用小语言模型的代码生成：对代码力量的深入评估 cs.SE

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2504.07343v1) [paper-pdf](http://arxiv.org/pdf/2504.07343v1)

**Authors**: Débora Souza, Rohit Gheyi, Lucas Albuquerque, Gustavo Soares, Márcio Ribeiro

**Abstract**: Large Language Models (LLMs) have demonstrated capabilities in code generation, potentially boosting developer productivity. However, their widespread adoption remains limited by high computational costs, significant energy demands, and security risks such as data leakage and adversarial attacks. As a lighter-weight alternative, Small Language Models (SLMs) offer faster inference, lower deployment overhead, and better adaptability to domain-specific tasks, making them an attractive option for real-world applications. While prior research has benchmarked LLMs on competitive programming tasks, such evaluations often focus narrowly on metrics like Elo scores or pass rates, overlooking deeper insights into model behavior, failure patterns, and problem diversity. Furthermore, the potential of SLMs to tackle complex tasks such as competitive programming remains underexplored. In this study, we benchmark five open SLMs - LLAMA 3.2 3B, GEMMA 2 9B, GEMMA 3 12B, DEEPSEEK-R1 14B, and PHI-4 14B - across 280 Codeforces problems spanning Elo ratings from 800 to 2100 and covering 36 distinct topics. All models were tasked with generating Python solutions. PHI-4 14B achieved the best performance among SLMs, with a pass@3 of 63.6%, approaching the proprietary O3-MINI-HIGH (86.8%). In addition, we evaluated PHI-4 14B on C++ and found that combining outputs from both Python and C++ increases its aggregated pass@3 to 73.6%. A qualitative analysis of PHI-4 14B's incorrect outputs revealed that some failures were due to minor implementation issues - such as handling edge cases or correcting variable initialization - rather than deeper reasoning flaws.

摘要: 大型语言模型（LLM）已经展示了代码生成的能力，这可能会提高开发人员的生产力。然而，它们的广泛采用仍然受到高计算成本、高能源需求以及数据泄露和对抗性攻击等安全风险的限制。作为一种轻量级的替代方案，小型语言模型（SLC）提供更快的推理、更低的部署负担以及对特定领域任务的更好的适应性，使其成为现实世界应用程序的有吸引力的选择。虽然之前的研究已经根据竞争性编程任务对LLM进行了基准测试，但此类评估通常狭隘地关注Elo分数或通过率等指标，而忽视了对模型行为、失败模式和问题多样性的更深入见解。此外，Slms解决竞争性编程等复杂任务的潜力仍然没有得到充分的开发。在这项研究中，我们对五个开放的LM进行了基准测试- LLAMA 3.2 3B、GEGMA 2 9 B、GEGMA 3 12 B、DEEPSEEK-R1 14 B和PHI-4 14 B-跨越280个Codeforce问题，涵盖Elo评级从800到2100，涵盖36个不同的主题。所有模型的任务都是生成Python解决方案。PHI-4 14 B实现了SLS中最好的性能，通过率@3为63.6%，接近专有的O3-MINI-HIGH（86.8%）。此外，我们在C++上评估了PHI-4 14 B，发现结合Python和C++的输出可以将其总通过率@3增加到73.6%。对PHI-4 14 B错误输出的定性分析显示，一些失败是由于小的实现问题（例如处理边缘情况或纠正变量初始化）而不是更深层次的推理缺陷。



## **43. LLM Safeguard is a Double-Edged Sword: Exploiting False Positives for Denial-of-Service Attacks**

LLM保障是一把双刃剑：利用假阳性进行拒绝服务攻击 cs.CR

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2410.02916v3) [paper-pdf](http://arxiv.org/pdf/2410.02916v3)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern for large language models (LLMs) in open deployment, motivating the development of safeguard methods that enforce ethical and responsible use through safety alignment or guardrail mechanisms. Jailbreak attacks that exploit the \emph{false negatives} of safeguard methods have emerged as a prominent research focus in the field of LLM security. However, we found that the malicious attackers could also exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a denial-of-service (DoS) affecting LLM users. To bridge the knowledge gap of this overlooked threat, we explore multiple attack methods that include inserting a short adversarial prompt into user prompt templates and corrupting the LLM on the server by poisoned fine-tuning. In both ways, the attack triggers safeguard rejections of user requests from the client. Our evaluation demonstrates the severity of this threat across multiple scenarios. For instance, in the scenario of white-box adversarial prompt injection, the attacker can use our optimization process to automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97% of user requests on Llama Guard 3. These findings reveal a new dimension in LLM safeguard evaluation -- adversarial robustness to false positives.

摘要: 安全性是开放部署中大型语言模型（LLM）的首要问题，这促使开发了通过安全对齐或护栏机制来执行道德和负责任使用的保障方法。利用安全措施的\“假阴性\”的越狱攻击已经成为LLM安全领域的一个突出的研究热点。然而，我们发现恶意攻击者也可以利用安全措施的误报，即，欺骗保护模型错误地阻止安全内容，导致影响LLM用户的拒绝服务（DoS）。为了弥合这个被忽视的威胁的知识差距，我们探索了多种攻击方法，包括在用户提示模板中插入简短的对抗提示，以及通过有毒微调损坏服务器上的LLM。通过这两种方式，攻击都会触发对客户端用户请求的安全拒绝。我们的评估表明了这种威胁在多种场景下的严重性。例如，在白盒对抗性提示注入的场景中，攻击者可以使用我们的优化过程自动生成看似安全的对抗性提示，大约只有30个字符长，普遍阻止Llama Guard 3上超过97%的用户请求。这些发现揭示了LLM保障评估的一个新维度--对假阳性的对抗稳健性。



## **44. SafeMLRM: Demystifying Safety in Multi-modal Large Reasoning Models**

SafeMLRM：揭开多模式大型推理模型中的安全性的神秘面纱 cs.LG

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2504.08813v1) [paper-pdf](http://arxiv.org/pdf/2504.08813v1)

**Authors**: Junfeng Fang, Yukai Wang, Ruipeng Wang, Zijun Yao, Kun Wang, An Zhang, Xiang Wang, Tat-Seng Chua

**Abstract**: The rapid advancement of multi-modal large reasoning models (MLRMs) -- enhanced versions of multimodal language models (MLLMs) equipped with reasoning capabilities -- has revolutionized diverse applications. However, their safety implications remain underexplored. While prior work has exposed critical vulnerabilities in unimodal reasoning models, MLRMs introduce distinct risks from cross-modal reasoning pathways. This work presents the first systematic safety analysis of MLRMs through large-scale empirical studies comparing MLRMs with their base MLLMs. Our experiments reveal three critical findings: (1) The Reasoning Tax: Acquiring reasoning capabilities catastrophically degrades inherited safety alignment. MLRMs exhibit 37.44% higher jailbreaking success rates than base MLLMs under adversarial attacks. (2) Safety Blind Spots: While safety degradation is pervasive, certain scenarios (e.g., Illegal Activity) suffer 25 times higher attack rates -- far exceeding the average 3.4 times increase, revealing scenario-specific vulnerabilities with alarming cross-model and datasets consistency. (3) Emergent Self-Correction: Despite tight reasoning-answer safety coupling, MLRMs demonstrate nascent self-correction -- 16.9% of jailbroken reasoning steps are overridden by safe answers, hinting at intrinsic safeguards. These findings underscore the urgency of scenario-aware safety auditing and mechanisms to amplify MLRMs' self-correction potential. To catalyze research, we open-source OpenSafeMLRM, the first toolkit for MLRM safety evaluation, providing unified interface for mainstream models, datasets, and jailbreaking methods. Our work calls for immediate efforts to harden reasoning-augmented AI, ensuring its transformative potential aligns with ethical safeguards.

摘要: 多模式大型推理模型（MLRM）--配备推理能力的多模式语言模型（MLRM）的增强版本--的快速发展彻底改变了各种应用。然而，它们的安全影响仍然没有得到充分的研究。虽然之前的工作暴露了单模式推理模型中的关键漏洞，但MLRM引入了跨模式推理路径的明显风险。这项工作通过比较MLRM与其基础MLLM的大规模实证研究，首次对MLRM进行了系统性安全性分析。我们的实验揭示了三个关键发现：（1）推理税：获得推理能力会灾难性地降低遗传安全一致性。在对抗性攻击下，MLRM的越狱成功率比基本MLLM高出37.44%。(2)安全盲点：虽然安全性下降普遍存在，但某些场景（例如，非法活动）遭受的攻击率高出25倍，远超过平均3.4倍的增幅，揭示了具有令人震惊的跨模型和数据集一致性的特定于集群的漏洞。(3)紧急自我纠正：尽管推理与答案的安全耦合紧密，但MLRM表现出了新生的自我纠正--16.9%的越狱推理步骤被安全答案推翻，暗示了内在的保障措施。这些发现强调了具有冲突意识的安全审计和增强MLRM自我纠正潜力的机制的紧迫性。为了促进研究，我们开源OpenSafeMLRM，这是第一个MLRM安全评估工具包，为主流模型、数据集和越狱方法提供统一的界面。我们的工作呼吁立即努力强化推理增强人工智能，确保其变革潜力与道德保障措施保持一致。



## **45. JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model**

JailDAM：使用视觉语言模型的自适应记忆的越狱检测 cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.03770v2) [paper-pdf](http://arxiv.org/pdf/2504.03770v2)

**Authors**: Yi Nian, Shenzhe Zhu, Yuehan Qin, Li Li, Ziyi Wang, Chaowei Xiao, Yue Zhao

**Abstract**: Multimodal large language models (MLLMs) excel in vision-language tasks but also pose significant risks of generating harmful content, particularly through jailbreak attacks. Jailbreak attacks refer to intentional manipulations that bypass safety mechanisms in models, leading to the generation of inappropriate or unsafe content. Detecting such attacks is critical to ensuring the responsible deployment of MLLMs. Existing jailbreak detection methods face three primary challenges: (1) Many rely on model hidden states or gradients, limiting their applicability to white-box models, where the internal workings of the model are accessible; (2) They involve high computational overhead from uncertainty-based analysis, which limits real-time detection, and (3) They require fully labeled harmful datasets, which are often scarce in real-world settings. To address these issues, we introduce a test-time adaptive framework called JAILDAM. Our method leverages a memory-based approach guided by policy-driven unsafe knowledge representations, eliminating the need for explicit exposure to harmful data. By dynamically updating unsafe knowledge during test-time, our framework improves generalization to unseen jailbreak strategies while maintaining efficiency. Experiments on multiple VLM jailbreak benchmarks demonstrate that JAILDAM delivers state-of-the-art performance in harmful content detection, improving both accuracy and speed.

摘要: 多模式大型语言模型（MLLM）在视觉语言任务中表现出色，但也存在生成有害内容的巨大风险，特别是通过越狱攻击。越狱攻击是指绕过模型中安全机制的故意操纵，导致生成不适当或不安全的内容。检测此类攻击对于确保负责任地部署MLLM至关重要。现有的越狱检测方法面临三个主要挑战：（1）许多方法依赖于模型隐藏状态或梯度，限制了其对白盒模型的适用性，而白盒模型的内部工作是可以访问的;（2）它们涉及基于不确定性的分析的高计算负担，这限制了实时检测，以及（3）它们需要完全标记的有害数据集，这在现实世界中通常是稀缺的。为了解决这些问题，我们引入了一个名为JAILDAM的测试时自适应框架。我们的方法利用了基于内存的方法，该方法由政策驱动的不安全知识表示指导，消除了显式暴露有害数据的需要。通过在测试期间动态更新不安全知识，我们的框架提高了对未见越狱策略的概括性，同时保持效率。多个VLM越狱基准测试的实验表明，JAILDAM在有害内容检测方面提供了最先进的性能，提高了准确性和速度。



## **46. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

StealthRank：通过StealthPropriation优化进行LLM排名操纵 cs.IR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05804v1) [paper-pdf](http://arxiv.org/pdf/2504.05804v1)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems.

摘要: 将大型语言模型（LLM）集成到信息检索系统中引入了新的攻击表面，特别是对于对抗性排名操纵。我们介绍了StealthRank，这是一种新型的对抗性排名攻击，它可以操纵LLM驱动的产品推荐系统，同时保持文本流畅性和隐蔽性。与经常引入可检测异常的现有方法不同，StealthRank采用基于能量的优化框架与Langevin动态相结合来生成StealthRank脚本（SPP）-嵌入产品描述中的对抗性文本序列，微妙而有效地影响LLM排名机制。我们在多个LLM中评估StealthRank，证明其能够秘密提高目标产品的排名，同时避免容易检测到的显式操纵痕迹。我们的结果表明，StealthRank在有效性和隐蔽性方面始终优于最先进的对抗排名基线，凸显了LLM驱动的推荐系统中的关键漏洞。



## **47. Separator Injection Attack: Uncovering Dialogue Biases in Large Language Models Caused by Role Separators**

分隔符注入攻击：揭露角色分隔符引起的大型语言模型中的对话偏见 cs.CL

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05689v1) [paper-pdf](http://arxiv.org/pdf/2504.05689v1)

**Authors**: Xitao Li, Haijun Wang, Jiang Wu, Ting Liu

**Abstract**: Conversational large language models (LLMs) have gained widespread attention due to their instruction-following capabilities. To ensure conversational LLMs follow instructions, role separators are employed to distinguish between different participants in a conversation. However, incorporating role separators introduces potential vulnerabilities. Misusing roles can lead to prompt injection attacks, which can easily misalign the model's behavior with the user's intentions, raising significant security concerns. Although various prompt injection attacks have been proposed, recent research has largely overlooked the impact of role separators on safety. This highlights the critical need to thoroughly understand the systemic weaknesses in dialogue systems caused by role separators. This paper identifies modeling weaknesses caused by role separators. Specifically, we observe a strong positional bias associated with role separators, which is inherent in the format of dialogue modeling and can be triggered by the insertion of role separators. We further develop the Separators Injection Attack (SIA), a new orthometric attack based on role separators. The experiment results show that SIA is efficient and extensive in manipulating model behavior with an average gain of 18.2% for manual methods and enhances the attack success rate to 100% with automatic methods.

摘要: 对话式大型语言模型（LLM）因其描述跟踪能力而受到广泛关注。为了确保对话LLM遵循说明，使用角色分隔符来区分对话中的不同参与者。然而，合并角色分隔符会带来潜在的漏洞。滥用角色可能会导致即时注入攻击，这很容易使模型的行为与用户的意图不一致，从而引发严重的安全问题。尽管人们提出了各种即时注射攻击，但最近的研究在很大程度上忽视了角色分离器对安全性的影响。这凸显了彻底了解角色分离造成的对话系统系统性弱点的迫切需要。本文指出了角色分隔符导致的建模弱点。具体来说，我们观察到与角色分隔符相关的强烈位置偏见，这是对话建模格式中固有的，可以通过插入角色分隔符来触发。我们进一步开发了分离器注入攻击（SIA），这是一种基于角色分离器的新的正向攻击。实验结果表明，SIA在操纵模型行为方面高效且广泛，手动方法的平均收益率为18.2%，自动方法的攻击成功率提高至100%。



## **48. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

糖衣毒药：良性一代解锁法学硕士越狱 cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05652v1) [paper-pdf](http://arxiv.org/pdf/2504.05652v1)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Jie-Zhang

**Abstract**: Large Language Models (LLMs) have become increasingly integral to a wide range of applications. However, they still remain the threat of jailbreak attacks, where attackers manipulate designed prompts to make the models elicit malicious outputs. Analyzing jailbreak methods can help us delve into the weakness of LLMs and improve it. In this paper, We reveal a vulnerability in large language models (LLMs), which we term Defense Threshold Decay (DTD), by analyzing the attention weights of the model's output on input and subsequent output on prior output: as the model generates substantial benign content, its attention weights shift from the input to prior output, making it more susceptible to jailbreak attacks. To demonstrate the exploitability of DTD, we propose a novel jailbreak attack method, Sugar-Coated Poison (SCP), which induces the model to generate substantial benign content through benign input and adversarial reasoning, subsequently producing malicious content. To mitigate such attacks, we introduce a simple yet effective defense strategy, POSD, which significantly reduces jailbreak success rates while preserving the model's generalization capabilities.

摘要: 大型语言模型（LLM）已经成为越来越广泛的应用程序的组成部分。然而，它们仍然是越狱攻击的威胁，攻击者操纵设计的提示，使模型引发恶意输出。分析越狱方法可以帮助我们深入研究LLM的弱点并对其进行改进。本文通过分析模型的输出对输入和后续输出对先前输出的注意力权重，揭示了大型语言模型（LLM）中的一个漏洞，我们称之为防御阈值衰减（DTD）：当模型生成大量良性内容时，其注意力权重从输入转移到先前输出，使其更容易受到越狱攻击。为了证明DTD的可利用性，我们提出了一种新的越狱攻击方法，糖衣毒药（SCP），它诱导模型通过良性输入和对抗性推理生成大量良性内容，随后产生恶意内容。为了减轻这种攻击，我们引入了一种简单而有效的防御策略POSD，它可以显着降低越狱成功率，同时保留模型的泛化能力。



## **49. SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments**

SceneRAP：针对现实世界环境中视觉语言模型的场景一致印刷对抗规划器 cs.CV

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2412.00114v2) [paper-pdf](http://arxiv.org/pdf/2412.00114v2)

**Authors**: Yue Cao, Yun Xing, Jie Zhang, Di Lin, Tianwei Zhang, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in interpreting visual content. While existing works demonstrate these models' vulnerability to deliberately placed adversarial texts, such texts are often easily identifiable as anomalous. In this paper, we present the first approach to generate scene-coherent typographic adversarial attacks that mislead advanced LVLMs while maintaining visual naturalness through the capability of the LLM-based agent. Our approach addresses three critical questions: what adversarial text to generate, where to place it within the scene, and how to integrate it seamlessly. We propose a training-free, multi-modal LLM-driven scene-coherent typographic adversarial planning (SceneTAP) that employs a three-stage process: scene understanding, adversarial planning, and seamless integration. The SceneTAP utilizes chain-of-thought reasoning to comprehend the scene, formulate effective adversarial text, strategically plan its placement, and provide detailed instructions for natural integration within the image. This is followed by a scene-coherent TextDiffuser that executes the attack using a local diffusion mechanism. We extend our method to real-world scenarios by printing and placing generated patches in physical environments, demonstrating its practical implications. Extensive experiments show that our scene-coherent adversarial text successfully misleads state-of-the-art LVLMs, including ChatGPT-4o, even after capturing new images of physical setups. Our evaluations demonstrate a significant increase in attack success rates while maintaining visual naturalness and contextual appropriateness. This work highlights vulnerabilities in current vision-language models to sophisticated, scene-coherent adversarial attacks and provides insights into potential defense mechanisms.

摘要: 大型视觉语言模型（LVLM）在解释视觉内容方面表现出了非凡的能力。虽然现有的作品证明了这些模型对故意放置的对抗文本的脆弱性，但此类文本通常很容易被识别为异常文本。在本文中，我们提出了第一种生成场景一致印刷对抗攻击的方法，这种攻击可以误导高级LVLM，同时通过基于LLM的代理的能力保持视觉自然性。我们的方法解决了三个关键问题：生成什么对抗文本、将其放置在场景中的位置以及如何无缝集成它。我们提出了一种免培训、多模式LLM驱动的场景一致印刷对抗性规划（SceneRAP），该规划采用三阶段流程：场景理解、对抗性规划和无缝集成。SceneRAP利用思想链推理来理解场景、制定有效的对抗文本、战略性地规划其放置，并为图像中的自然整合提供详细的说明。随后是场景一致的文本扩散用户，它使用本地扩散机制执行攻击。我们通过打印并将生成的补丁放置在物理环境中，将我们的方法扩展到现实世界场景，展示其实际含义。大量实验表明，即使在捕获物理设置的新图像之后，我们的场景连贯对抗文本也能成功误导最先进的LVLM，包括ChatGPT-4 o。我们的评估表明，攻击成功率显着提高，同时保持视觉自然性和上下文适当性。这项工作强调了当前视觉语言模型对复杂、场景一致的对抗攻击的脆弱性，并提供了对潜在防御机制的见解。



## **50. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

揭示一致大型语言模型内在的道德脆弱性 cs.CL

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05050v1) [paper-pdf](http://arxiv.org/pdf/2504.05050v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.

摘要: 大型语言模型（LLM）是人工通用智能的基础探索，但它们通过指令调整和偏好学习与人类价值观的一致只能实现表面的合规性。在这里，我们证明，预训练期间嵌入的有害知识在LLM参数记忆中作为不可磨灭的“黑暗模式”持续存在，逃避对齐保障措施，并在分布变化时的对抗诱导下重新浮出水面。在这项研究中，我们首先通过证明当前的对齐方法只产生知识集合中的局部“安全区域”来从理论上分析对齐LLM的内在道德脆弱性。相比之下，预先训练的知识仍然通过高可能性的对抗轨迹与有害概念保持全球联系。基于这一理论见解，我们通过在分布转移下采用语义一致诱导来从经验上验证我们的发现--一种通过优化的对抗提示系统性地绕过对齐约束的方法。这种理论和经验相结合的方法在23个最先进的对齐LLM中的19个（包括DeepSeek-R1和LLaMA-3）上实现了100%的攻击成功率，揭示了它们的普遍漏洞。




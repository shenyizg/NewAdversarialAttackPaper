# Latest Large Language Model Attack Papers
**update at 2025-10-11 09:49:51**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AutoRed: A Free-form Adversarial Prompt Generation Framework for Automated Red Teaming**

AutoRed：一个用于自动化红色团队的自由形式对抗提示生成框架 cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08329v1) [paper-pdf](http://arxiv.org/pdf/2510.08329v1)

**Authors**: Muxi Diao, Yutao Mou, Keqing He, Hanbo Song, Lulu Zhao, Shikun Zhang, Wei Ye, Kongming Liang, Zhanyu Ma

**Abstract**: The safety of Large Language Models (LLMs) is crucial for the development of trustworthy AI applications. Existing red teaming methods often rely on seed instructions, which limits the semantic diversity of the synthesized adversarial prompts. We propose AutoRed, a free-form adversarial prompt generation framework that removes the need for seed instructions. AutoRed operates in two stages: (1) persona-guided adversarial instruction generation, and (2) a reflection loop to iteratively refine low-quality prompts. To improve efficiency, we introduce a verifier to assess prompt harmfulness without querying the target models. Using AutoRed, we build two red teaming datasets -- AutoRed-Medium and AutoRed-Hard -- and evaluate eight state-of-the-art LLMs. AutoRed achieves higher attack success rates and better generalization than existing baselines. Our results highlight the limitations of seed-based approaches and demonstrate the potential of free-form red teaming for LLM safety evaluation. We will open source our datasets in the near future.

摘要: 大型语言模型（LLM）的安全性对于开发值得信赖的人工智能应用程序至关重要。现有的红色分组方法通常依赖于种子指令，这限制了合成对抗提示的语义多样性。我们提出AutoRed，这是一个自由形式的对抗性提示生成框架，它消除了对种子指令的需要。AutoRed分两个阶段运行：（1）角色引导的对抗指令生成，和（2）迭代地细化低质量提示的反射循环。为了提高效率，我们引入了一个验证器来评估即时危害性，而无需查询目标模型。使用AutoRed，我们构建了两个红色团队数据集-- AutoRed-Medium和AutoRed-Hard --并评估了八个最先进的LLM。AutoRed比现有基线实现了更高的攻击成功率和更好的概括性。我们的结果强调了基于种子的方法的局限性，并展示了自由形式的红色团队在LLM安全性评估中的潜力。我们将在不久的将来开放我们的数据集。



## **2. MCPSecBench: A Systematic Security Benchmark and Playground for Testing Model Context Protocols**

MCPSecBench：测试模型上下文协议的系统安全基准和游乐场 cs.CR

This is a technical report from Lingnan University, Hong Kong. Code  is available at https://github.com/AIS2Lab/MCPSecBench

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2508.13220v2) [paper-pdf](http://arxiv.org/pdf/2508.13220v2)

**Authors**: Yixuan Yang, Daoyuan Wu, Yufan Chen

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications via the Model Context Protocol (MCP), a universal, open standard for connecting AI agents with data sources and external tools. While MCP enhances the capabilities of LLM-based agents, it also introduces new security risks and expands their attack surfaces. In this paper, we present the first systematic taxonomy of MCP security, identifying 17 attack types across 4 primary attack surfaces. We introduce MCPSecBench, a comprehensive security benchmark and playground that integrates prompt datasets, MCP servers, MCP clients, attack scripts, and protection mechanisms to evaluate these attacks across three major MCP providers. Our benchmark is modular and extensible, allowing researchers to incorporate custom implementations of clients, servers, and transport protocols for systematic security assessment. Experimental results show that over 85% of the identified attacks successfully compromise at least one platform, with core vulnerabilities universally affecting Claude, OpenAI, and Cursor, while prompt-based and tool-centric attacks exhibit considerable variability across different hosts and models. In addition, current protection mechanisms have little effect against these attacks. Overall, MCPSecBench standardizes the evaluation of MCP security and enables rigorous testing across all MCP layers.

摘要: 大型语言模型（LLM）通过模型上下文协议（HCP）越来越多地集成到现实世界的应用程序中，模型上下文协议（HCP）是一种通用的开放标准，用于连接人工智能代理与数据源和外部工具。虽然HCP增强了基于LLM的代理的能力，但它也引入了新的安全风险并扩大了其攻击面。在本文中，我们提出了第一个系统性的LCP安全分类，识别了4个主要攻击表面的17种攻击类型。我们引入了MCPSecBench，这是一个全面的安全基准和游乐场，集成了提示数据集、HCP服务器、HCP客户端、攻击脚本和保护机制，以评估三大主要HCP提供商之间的这些攻击。我们的基准是模块化和可扩展的，允许研究人员整合客户端、服务器和传输协议的自定义实现，以进行系统性安全评估。实验结果表明，超过85%的已识别攻击成功危害至少一个平台，核心漏洞普遍影响Claude、OpenAI和Cursor，而基于预算和以工具为中心的攻击在不同的主机和模型中表现出相当大的变化性。此外，当前的保护机制对这些攻击几乎没有效果。总体而言，MCPSecBench实现了对LCP安全性的评估，并实现了对所有LCP层的严格测试。



## **3. Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning**

注意步骤：LLM微调后激活的休眠对抗行为 cs.LG

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2505.16567v3) [paper-pdf](http://arxiv.org/pdf/2505.16567v3)

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev

**Abstract**: Finetuning open-weight Large Language Models (LLMs) is standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets leads to predictable behaviors. In this paper, we demonstrate, for the first time, that an adversary can create compromised LLMs that are performant and benign, yet exhibit adversarial behaviors once finetuned by downstream users. To this end, we propose an attack, FAB (Finetuning-activated Adversarial Behaviors), which compromises an LLM via meta-learning techniques that simulate downstream finetuning, explicitly optimizing for the emergence of adversarial behaviors in the finetuned models. At the same time, the compromised LLM is regularized to retain general capabilities and to exhibit no adversarial behaviors prior to finetuning. As a result, when users finetune (e.g., instruction-tuning, distillation, DPO) the seemingly benign model on their own datasets, they unknowingly trigger its dormant adversarial behavior. We experimentally demonstrate the effectiveness of FAB across multiple LLMs and three commonly considered target behaviors: unsolicited advertising, jailbreakability, and over-refusal. We show that FAB-triggers are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler, post-training algorithm). Our findings challenge prevailing assumptions on the security of finetuning, revealing a critical attack vector.

摘要: 微调开权重大型语言模型（LLM）是实现特定任务性能改进的标准实践。到目前为止，微调一直被认为是一个受控且安全的过程，其中对良性数据集的训练会导致可预测的行为。在本文中，我们首次证明，对手可以创建高性能且良性的受损LLM，但一旦被下游用户微调，就会表现出对抗行为。为此，我们提出了一种攻击FAB（微调激活的对抗行为），它通过模拟下游微调的元学习技术来损害LLM，明确优化微调模型中对抗行为的出现。与此同时，受损的LLM被规范化，以保留一般能力，并且在微调之前不表现出对抗行为。因此，当用户微调（例如，描述-调优、蒸馏、DPO）在他们自己的数据集上看似良性的模型，但他们在不知不觉中触发了其休眠的对抗行为。我们通过实验证明了FAB在多个LLM和三种常见的目标行为中的有效性：未经请求的广告、越狱和过度拒绝。我们表明FAB触发器对用户做出的各种微调选择是稳健的（例如，数据集、步骤数、调度器、训练后算法）。我们的发现挑战了有关微调安全性的普遍假设，揭示了一个关键的攻击载体。



## **4. Chain-of-Trigger: An Agentic Backdoor that Paradoxically Enhances Agentic Robustness**

触发链：一个反常地增强显着稳健性的显着后门 cs.AI

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08238v1) [paper-pdf](http://arxiv.org/pdf/2510.08238v1)

**Authors**: Jiyang Qiu, Xinbei Ma, Yunqing Xu, Zhuosheng Zhang, Hai Zhao

**Abstract**: The rapid deployment of large language model (LLM)-based agents in real-world applications has raised serious concerns about their trustworthiness. In this work, we reveal the security and robustness vulnerabilities of these agents through backdoor attacks. Distinct from traditional backdoors limited to single-step control, we propose the Chain-of-Trigger Backdoor (CoTri), a multi-step backdoor attack designed for long-horizon agentic control. CoTri relies on an ordered sequence. It starts with an initial trigger, and subsequent ones are drawn from the environment, allowing multi-step manipulation that diverts the agent from its intended task. Experimental results show that CoTri achieves a near-perfect attack success rate (ASR) while maintaining a near-zero false trigger rate (FTR). Due to training data modeling the stochastic nature of the environment, the implantation of CoTri paradoxically enhances the agent's performance on benign tasks and even improves its robustness against environmental distractions. We further validate CoTri on vision-language models (VLMs), confirming its scalability to multimodal agents. Our work highlights that CoTri achieves stable, multi-step control within agents, improving their inherent robustness and task capabilities, which ultimately makes the attack more stealthy and raises potential safty risks.

摘要: 基于大型语言模型（LLM）的代理在现实世界应用程序中的快速部署引发了人们对其可信度的严重担忧。在这项工作中，我们通过后门攻击揭示了这些代理的安全性和鲁棒性漏洞。与仅限于一步控制的传统后门不同，我们提出了触发链后门（CoTri），这是一种旨在长视野代理控制的多步后门攻击。CoTri依赖于有序序列。它以初始触发器开始，随后的触发器从环境中提取，允许多步操作，将代理从其预期任务中转移。实验结果表明，CoTri实现了接近完美的攻击成功率（ASR），同时保持了接近零的错误触发率（FTR）。由于训练数据模拟了环境的随机性，CoTri的植入反而增强了智能体在良性任务上的性能，甚至提高了其对环境干扰的鲁棒性。我们进一步验证CoTri的视觉语言模型（VLM），确认其可扩展性多模态代理。我们的工作强调，CoTri在代理中实现了稳定的多步控制，提高了它们固有的鲁棒性和任务能力，最终使攻击更加隐蔽，并增加了潜在的安全风险。



## **5. Defending MoE LLMs against Harmful Fine-Tuning via Safety Routing Alignment**

通过安全路由对准保护MoE LLM免受有害微调 cs.CR

Under review

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2509.22745v2) [paper-pdf](http://arxiv.org/pdf/2509.22745v2)

**Authors**: Jaehan Kim, Minkyoo Song, Seungwon Shin, Sooel Son

**Abstract**: Recent large language models (LLMs) have increasingly adopted the Mixture-of-Experts (MoE) architecture for efficiency. MoE-based LLMs heavily depend on a superficial safety mechanism in which harmful inputs are routed safety-critical experts. However, our analysis reveals that routing decisions for harmful inputs drift significantly after fine-tuning, exposing a critical vulnerability to harmful fine-tuning (HFT) attacks. Existing defenses, primarily designed for monolithic LLMs, are less effective for MoE LLMs as they fail to prevent drift in harmful input routing. To address this limitation, we propose SafeMoE, a safe fine-tuning method tailored to MoE LLMs. SafeMoE directly mitigates routing drift by penalizing the gap between the routing weights of a fine-tuned model and those of the initial safety-aligned model, thereby preserving the safety-aligned routing of harmful inputs to safety-critical experts. Experiments on open-source MoE LLMs ranging from 7B to 141B parameters demonstrate that SafeMoE effectively mitigates HFT attacks, reducing the harmfulness score of OLMoE from 62.0 to 5.0, for example, while maintaining task utility within 1% degradation and incurring only 2% overhead. It significantly outperforms state-of-the-art defense methods for safeguarding LLM fine-tuning and remains effective in recent large-scale MoE LLMs such as gpt-oss and Llama 4. Our implementation is available at https://anonymous.4open.science/r/SafeMoE.

摘要: 为了提高效率，最近的大型语言模型（LLM）越来越多地采用专家混合（MoE）架构。基于教育部的LLM严重依赖于肤浅的安全机制，其中有害的输入由安全关键专家发送。然而，我们的分析表明，微调后有害输入的路由决策会显着漂移，从而暴露了有害微调（HFT）攻击的关键漏洞。现有的防御主要为单片LLM设计，但对于MoE LLM来说效果较差，因为它们无法防止有害输入路由的漂移。为了解决这一限制，我们提出了SafeMoE，这是一种针对MoE LLM量身定制的安全微调方法。SafeMoE通过惩罚微调模型的路由权重与初始安全对齐模型的路由权重之间的差距来直接减轻路由漂移，从而为安全关键专家保留有害输入的安全对齐路由。在7 B至141 B参数范围内的开源MoE LLM上的实验表明，SafeMoE有效地减轻了HFT攻击，例如将OLMoE的危害性评分从62.0降低到5.0，同时将任务效用保持在1%的降级范围内，仅产生2%的额外费用。它在保护LLM微调方面的性能明显优于最先进的防御方法，并且在最近的大规模MoE LLM（例如gtt-oss和Llama 4）中仍然有效。我们的实施可在https://anonymous.4open.science/r/SafeMoE上获取。



## **6. Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs**

多触发中毒放大了LLM中的后门漏洞 cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2507.11112v2) [paper-pdf](http://arxiv.org/pdf/2507.11112v2)

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.

摘要: 最近的研究表明，大型语言模型（LLM）很容易受到数据中毒攻击，其中恶意训练示例嵌入了由特定输入模式触发的隐藏行为。然而，大多数现有的作品假设一个短语并关注攻击的有效性，对触发机制以及多个触发如何在模型内相互作用的理解有限。本文中，我们提出了一个研究LLM中毒的框架。我们表明，多个不同的后门触发器可以在单个模型中共存，而不会相互干扰，从而使对手能够同时嵌入多个触发器。使用具有高嵌入相似性的多个触发器，我们证明即使令牌被长令牌跨度替换或分开，中毒触发器也可以实现稳健的激活。我们的研究结果揭示了LLC中更广泛、更持久的脆弱性表面。为了减轻这种威胁，我们提出了一种事后恢复方法，该方法根据分层权重差异分析选择性地重新训练特定的模型组件。我们的方法通过最少的参数更新有效地消除了触发行为，从而提供了针对多触发中毒的实用有效防御。



## **7. Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks**

打破评论者：评估文本对抗攻击下自动同行评审中大型语言模型的脆弱性 cs.CL

Minor correction: Fixed sign errors in the results table. The update  does not affect the main findings or conclusions

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2506.11113v3) [paper-pdf](http://arxiv.org/pdf/2506.11113v3)

**Authors**: Tzu-Ling Lin, Wei-Chih Chen, Teng-Fang Hsiao, Hou-I Liu, Ya-Hsin Yeh, Yu Kai Chan, Wen-Sheng Lien, Po-Yen Kuo, Philip S. Yu, Hong-Han Shuai

**Abstract**: Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication.

摘要: 同行评审对于保持学术质量至关重要，但提交量的增加给评审者带来了沉重的负担。大型语言模型（LLM）在此过程中提供了潜在的帮助，但它们对文本对抗攻击的敏感性引发了可靠性问题。本文研究了在存在此类攻击的情况下用作自动审查员的LLM的稳健性。我们重点关注三个关键问题：（1）与人类评审员相比，LLM在生成评审方面的有效性。(2)对抗性攻击对LLM生成的评论的可靠性的影响。(3)LLM为基础的审查的挑战和潜在的缓解策略。我们的评估揭示了重大的漏洞，因为文本操作可能会扭曲LLM评估。我们提供了一个全面的评估LLM性能的自动同行评审，并分析其对抗攻击的鲁棒性。我们的研究结果强调了解决对抗风险的重要性，以确保人工智能加强而不是损害学术交流的完整性。



## **8. DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm**

DNA-DetectLLM：通过DNA启发的突变修复范式揭示人工智能生成的文本 cs.CL

NeurIPS 2025 Spotlight

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2509.15550v2) [paper-pdf](http://arxiv.org/pdf/2509.15550v2)

**Authors**: Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang, Yanan Cao

**Abstract**: The rapid advancement of large language models (LLMs) has blurred the line between AI-generated and human-written text. This progress brings societal risks such as misinformation, authorship ambiguity, and intellectual property concerns, highlighting the urgent need for reliable AI-generated text detection methods. However, recent advances in generative language modeling have resulted in significant overlap between the feature distributions of human-written and AI-generated text, blurring classification boundaries and making accurate detection increasingly challenging. To address the above challenges, we propose a DNA-inspired perspective, leveraging a repair-based process to directly and interpretably capture the intrinsic differences between human-written and AI-generated text. Building on this perspective, we introduce DNA-DetectLLM, a zero-shot detection method for distinguishing AI-generated and human-written text. The method constructs an ideal AI-generated sequence for each input, iteratively repairs non-optimal tokens, and quantifies the cumulative repair effort as an interpretable detection signal. Empirical evaluations demonstrate that our method achieves state-of-the-art detection performance and exhibits strong robustness against various adversarial attacks and input lengths. Specifically, DNA-DetectLLM achieves relative improvements of 5.55% in AUROC and 2.08% in F1 score across multiple public benchmark datasets. Code and data are available at https://github.com/Xiaoweizhu57/DNA-DetectLLM.

摘要: 大型语言模型（LLM）的快速发展模糊了人工智能生成的文本和人类编写的文本之间的界限。这一进展带来了错误信息、作者身份模糊和知识产权问题等社会风险，凸显了对可靠的人工智能生成文本检测方法的迫切需求。然而，生成式语言建模的最新进展导致人类书写文本和人工智能生成文本的特征分布之间存在显着重叠，模糊了分类边界，并使准确检测变得越来越具有挑战性。为了解决上述挑战，我们提出了一种受DNA启发的视角，利用基于修复的流程来直接且可解释地捕捉人类书写和人工智能生成的文本之间的内在差异。基于这一观点，我们引入了DNA-DetectLLM，这是一种用于区分人工智能生成文本和人类编写文本的零镜头检测方法。该方法为每个输入构建理想的人工智能生成序列，迭代地修复非最优令牌，并将累积修复工作量化为可解释的检测信号。经验评估表明，我们的方法实现了最先进的检测性能，并对各种对抗攻击和输入长度表现出强大的鲁棒性。具体来说，在多个公共基准数据集中，DNA-DetectLLM在AUROC和F1评分上相对提高了5.55%，在F1评分上相对提高了2.08%。代码和数据可在https://github.com/Xiaoweizhu57/DNA-DetectLLM上获取。



## **9. (Token-Level) InfoRMIA: Stronger Membership Inference and Memorization Assessment for LLMs**

（令牌级）InfoRMIA：更强的LLM成员资格推断和认证评估 cs.LG

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.05582v2) [paper-pdf](http://arxiv.org/pdf/2510.05582v2)

**Authors**: Jiashu Tao, Reza Shokri

**Abstract**: Machine learning models are known to leak sensitive information, as they inevitably memorize (parts of) their training data. More alarmingly, large language models (LLMs) are now trained on nearly all available data, which amplifies the magnitude of information leakage and raises serious privacy risks. Hence, it is more crucial than ever to quantify privacy risk before the release of LLMs. The standard method to quantify privacy is via membership inference attacks, where the state-of-the-art approach is the Robust Membership Inference Attack (RMIA). In this paper, we present InfoRMIA, a principled information-theoretic formulation of membership inference. Our method consistently outperforms RMIA across benchmarks while also offering improved computational efficiency.   In the second part of the paper, we identify the limitations of treating sequence-level membership inference as the gold standard for measuring leakage. We propose a new perspective for studying membership and memorization in LLMs: token-level signals and analyses. We show that a simple token-based InfoRMIA can pinpoint which tokens are memorized within generated outputs, thereby localizing leakage from the sequence level down to individual tokens, while achieving stronger sequence-level inference power on LLMs. This new scope rethinks privacy in LLMs and can lead to more targeted mitigation, such as exact unlearning.

摘要: 众所周知，机器学习模型会泄露敏感信息，因为它们不可避免地记住（部分）训练数据。更令人担忧的是，大型语言模型（LLM）现在几乎是在所有可用数据上训练的，这放大了信息泄露的程度并引发了严重的隐私风险。因此，在LLM发布之前量化隐私风险比以往任何时候都更加重要。量化隐私的标准方法是通过成员资格推理攻击，其中最先进的方法是鲁棒成员资格推理攻击（RMIA）。在本文中，我们介绍了InfoRMIA，这是隶属推理的一种原则性信息论公式。我们的方法在各个基准测试中始终优于RMIA，同时还提供了更高的计算效率。   在论文的第二部分中，我们确定了将序列水平隶属度推断作为测量泄漏的金标准的局限性。我们提出了一个研究LLM中的成员资格和记忆的新视角：代币级信号和分析。我们发现，一个简单的基于令牌的InfoRMIA可以精确定位哪些令牌被存储在生成的输出中，从而将泄漏从序列级定位到单个令牌，同时在LLM上实现更强的序列级推理能力。这个新的范围重新考虑了LLM中的隐私，并可以导致更有针对性的缓解，例如精确的遗忘。



## **10. Fewer Weights, More Problems: A Practical Attack on LLM Pruning**

更少的权重，更多的问题：对LLM修剪的实用攻击 cs.LG

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07985v1) [paper-pdf](http://arxiv.org/pdf/2510.07985v1)

**Authors**: Kazuki Egashira, Robin Staab, Thibaud Gloaguen, Mark Vero, Martin Vechev

**Abstract**: Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference. Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed. While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored. In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited. In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors. Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned. With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned. Then, they can repair the model by using parameters that are likely to be pruned, effectively canceling out the injected behavior in the unpruned model. We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to $95.7\%$ for jailbreak, $98.7\%$ for benign instruction refusal, and $99.5\%$ for targeted content injection). Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression.

摘要: 模型修剪，即删除模型权重的子集已成为减少推理期间大型语言模型（LLM）内存占用的一种主要方法。值得注意的是，vLLM等流行推理引擎使用户能够在部署下载的模型之前方便地修剪它们。虽然修剪方法的实用性和效率有了显着提高，但修剪的安全影响仍然没有得到充分的研究。在这项工作中，我们首次表明现代LLM修剪方法可以被恶意利用。特别是，对手可以构建一个看起来良性但一旦修剪，就会表现出恶意行为的模型。我们的方法基于这样的想法：对手可以计算代理指标，该指标估计每个参数被修剪的可能性。有了这些信息，对手可以首先将恶意行为注入到那些不太可能被修剪的参数中。然后，他们可以通过使用可能被修剪的参数来修复模型，从而有效地抵消未修剪模型中注入的行为。我们通过对五个模型的广泛评估来证明攻击的严重性;应用vLLM中的任何修剪（Magnitude、Wanda和SparseGPT）后，它在各种攻击场景中始终表现出强烈的恶意行为（越狱成功率高达95.7美元，良性指令拒绝成功率高达98.7美元，定向内容注入成功率高达99.5美元）。我们的结果揭示了一个关键的部署时安全差距，并强调了模型压缩中迫切需要更强的安全意识。



## **11. Fine-Tuning Jailbreaks under Highly Constrained Black-Box Settings: A Three-Pronged Approach**

在高度限制的黑匣子环境下微调越狱：三角方法 cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.01342v2) [paper-pdf](http://arxiv.org/pdf/2510.01342v2)

**Authors**: Xiangfang Li, Yu Wang, Bo Li

**Abstract**: With the rapid advancement of large language models (LLMs), ensuring their safe use becomes increasingly critical. Fine-tuning is a widely used method for adapting models to downstream tasks, yet it is vulnerable to jailbreak attacks. However, most existing studies focus on overly simplified attack scenarios, limiting their practical relevance to real-world defense settings. To make this risk concrete, we present a three-pronged jailbreak attack and evaluate it against provider defenses under a dataset-only black-box fine-tuning interface. In this setting, the attacker can only submit fine-tuning data to the provider, while the provider may deploy defenses across stages: (1) pre-upload data filtering, (2) training-time defensive fine-tuning, and (3) post-training safety audit. Our attack combines safety-styled prefix/suffix wrappers, benign lexical encodings (underscoring) of sensitive tokens, and a backdoor mechanism, enabling the model to learn harmful behaviors while individual datapoints appear innocuous. Extensive experiments demonstrate the effectiveness of our approach. In real-world deployment, our method successfully jailbreaks GPT-4.1 and GPT-4o on the OpenAI platform with attack success rates above 97% for both models. Our code is available at https://github.com/lxf728/tri-pronged-ft-attack.

摘要: 随着大型语言模型（LLM）的迅速发展，确保它们的安全使用变得越来越重要。微调是一种广泛使用的方法，用于使模型适应下游任务，但它很容易受到越狱攻击。然而，大多数现有的研究都集中在过于简化的攻击场景上，限制了它们与现实世界防御环境的实际相关性。为了使这一风险具体化，我们提出了一种三管齐下的越狱攻击，并在仅限厕所的黑匣子微调界面下针对提供商防御进行评估。在这种设置下，攻击者只能向提供者提交微调数据，而提供者可以跨阶段部署防御：（1）上传前数据过滤，（2）培训时防御微调，（3）培训后安全审计。我们的攻击结合了安全风格的前置/后缀包装器、敏感令牌的良性词汇编码（强调）和后门机制，使模型能够学习有害行为，而单个数据点看起来无害。大量的实验证明了我们方法的有效性。在现实世界的部署中，我们的方法在OpenAI平台上成功越狱了GPT-4.1和GPT-4 o，两种模型的攻击成功率均超过97%。我们的代码可在https://github.com/lxf728/tri-pronged-ft-attack上获取。



## **12. Rule Encoding and Compliance in Large Language Models: An Information-Theoretic Analysis**

大型语言模型中的规则编码与合规性：信息论分析 cs.AI

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.05106v2) [paper-pdf](http://arxiv.org/pdf/2510.05106v2)

**Authors**: Joachim Diederich

**Abstract**: The design of safety-critical agents based on large language models (LLMs) requires more than simple prompt engineering. This paper presents a comprehensive information-theoretic analysis of how rule encodings in system prompts influence attention mechanisms and compliance behaviour. We demonstrate that rule formats with low syntactic entropy and highly concentrated anchors reduce attention entropy and improve pointer fidelity, but reveal a fundamental trade-off between anchor redundancy and attention entropy that previous work failed to recognize. Through formal analysis of multiple attention architectures including causal, bidirectional, local sparse, kernelized, and cross-attention mechanisms, we establish bounds on pointer fidelity and show how anchor placement strategies must account for competing fidelity and entropy objectives. Combining these insights with a dynamic rule verification architecture, we provide a formal proof that hot reloading of verified rule sets increases the asymptotic probability of compliant outputs. These findings underscore the necessity of principled anchor design and dual enforcement mechanisms to protect LLM-based agents against prompt injection attacks while maintaining compliance in evolving domains.

摘要: 基于大型语言模型（LLM）的安全关键代理的设计需要的不仅仅是简单的即时工程。本文对系统提示中的规则编码如何影响注意机制和合规行为进行了全面的信息论分析。我们证明，具有低语法熵和高度集中的锚点的规则格式可以减少注意力熵并提高指针保真度，但揭示了锚点冗余和注意力熵之间的基本权衡，而之前的工作未能认识到这一点。通过对多个注意力架构（包括因果、双向、局部稀疏、核化和交叉注意力机制）的形式分析，我们建立了指针保真度的界限，并展示了锚放置策略必须如何考虑竞争的保真度和熵目标。将这些见解与动态规则验证架构相结合，我们提供了一个正式证明，证明已验证的规则集的热重新加载增加了合规输出的渐进概率。这些发现强调了有原则的锚设计和双重执行机制的必要性，以保护基于LLM的代理免受即时注入攻击，同时保持不断发展的领域的合规性。



## **13. PiCo: Jailbreaking Multimodal Large Language Models via Pictorial Code Contextualization**

PiCo：通过图形代码上下文化越狱多模态大型语言模型 cs.CR

Accepted to IEEE International Conference on Multimedia and Expo  (ICME) 2025

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2504.01444v4) [paper-pdf](http://arxiv.org/pdf/2504.01444v4)

**Authors**: Aofan Liu, Lulu Tang, Ting Pan, Yuguo Yin, Bin Wang, Ao Yang

**Abstract**: Multimodal Large Language Models (MLLMs), which integrate vision and other modalities into Large Language Models (LLMs), significantly enhance AI capabilities but also introduce new security vulnerabilities. By exploiting the vulnerabilities of the visual modality and the long-tail distribution characteristic of code training data, we present PiCo, a novel jailbreaking framework designed to progressively bypass multi-tiered defense mechanisms in advanced MLLMs. PiCo employs a tier-by-tier jailbreak strategy, using token-level typographic attacks to evade input filtering and embedding harmful intent within programming context instructions to bypass runtime monitoring. To comprehensively assess the impact of attacks, a new evaluation metric is further proposed to assess both the toxicity and helpfulness of model outputs post-attack. By embedding harmful intent within code-style visual instructions, PiCo achieves an average Attack Success Rate (ASR) of 84.13% on Gemini-Pro Vision and 52.66% on GPT-4, surpassing previous methods. Experimental results highlight the critical gaps in current defenses, underscoring the need for more robust strategies to secure advanced MLLMs.

摘要: 多模式大型语言模型（MLLM）将视觉和其他模式集成到大型语言模型（LLM）中，显着增强了人工智能能力，但也引入了新的安全漏洞。通过利用视觉模式的漏洞和代码训练数据的长尾分布特征，我们提出了PiCo，这是一种新型越狱框架，旨在逐步绕过高级MLLM中的多层防御机制。PiCo采用逐层越狱策略，使用标记级印刷攻击来逃避输入过滤，并在编程上下文指令中嵌入有害意图以绕过运行时监控。为了全面评估攻击的影响，进一步提出了一种新的评估指标来评估攻击后模型输出的毒性和帮助性。通过在代码风格的视觉指令中嵌入有害意图，PiCo在Gemini-Pro Vision上实现了84.13%的平均攻击成功率（ASB），在GPT-4上实现了52.66%的平均攻击成功率（ASB），超过了之前的方法。实验结果凸显了当前防御中的关键差距，强调需要更稳健的策略来保护高级MLLM。



## **14. Logic Jailbreak: Efficiently Unlocking LLM Safety Restrictions Through Formal Logical Expression**

逻辑越狱：通过形式逻辑表达有效解锁LLM安全限制 cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2505.13527v2) [paper-pdf](http://arxiv.org/pdf/2505.13527v2)

**Authors**: Jingyu Peng, Maolin Wang, Nan Wang, Jiatong Li, Yuchen Li, Yuyang Ye, Wanyu Wang, Pengyue Jia, Kai Zhang, Xiangyu Zhao

**Abstract**: Despite substantial advancements in aligning large language models (LLMs) with human values, current safety mechanisms remain susceptible to jailbreak attacks. We hypothesize that this vulnerability stems from distributional discrepancies between alignment-oriented prompts and malicious prompts. To investigate this, we introduce LogiBreak, a novel and universal black-box jailbreak method that leverages logical expression translation to circumvent LLM safety systems. By converting harmful natural language prompts into formal logical expressions, LogiBreak exploits the distributional gap between alignment data and logic-based inputs, preserving the underlying semantic intent and readability while evading safety constraints. We evaluate LogiBreak on a multilingual jailbreak dataset spanning three languages, demonstrating its effectiveness across various evaluation settings and linguistic contexts.

摘要: 尽管在将大型语言模型（LLM）与人类价值观结合方面取得了长足的进步，但当前的安全机制仍然容易受到越狱攻击。我们假设此漏洞源于面向提示的提示和恶意提示之间的分布差异。为了研究这一点，我们引入了LogiBreak，这是一种新颖且通用的黑匣子越狱方法，它利用逻辑表达翻译来规避LLM安全系统。通过将有害的自然语言提示转换为形式的逻辑表达，LogiBreak利用了对齐数据和基于逻辑的输入之间的分布差距，保留了底层的语义意图和可读性，同时规避了安全限制。我们在跨越三种语言的多语言越狱数据集上评估LogiBreak，展示了其在各种评估环境和语言背景下的有效性。



## **15. MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation**

MetaDefense：在生成之前和生成期间防御基于Finetuning的越狱攻击 cs.LG

Accepted By NeurIPS 2025

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07835v1) [paper-pdf](http://arxiv.org/pdf/2510.07835v1)

**Authors**: Weisen Jiang, Sinno Jialin Pan

**Abstract**: This paper introduces MetaDefense, a novel framework for defending against finetuning-based jailbreak attacks in large language models (LLMs). We observe that existing defense mechanisms fail to generalize to harmful queries disguised by unseen attack templates, despite LLMs being capable of distinguishing disguised harmful queries in the embedding space. Based on these insights, we propose a two-stage defense approach: (i) pre-generation defense that detects harmful queries before response generation begins, and (ii) mid-generation defense that monitors partial responses during generation to prevent outputting more harmful content. Our MetaDefense trains the LLM to predict the harmfulness of both queries and partial responses using specialized prompts, enabling early termination of potentially harmful interactions. Extensive experiments across multiple LLM architectures (LLaMA-2-7B, Qwen-2.5-3B-Instruct, and LLaMA-3.2-3B-Instruct) demonstrate that MetaDefense significantly outperforms existing defense mechanisms, achieving robust defense against harmful queries with seen and unseen attack templates while maintaining competitive performance on benign tasks. Code is available at https://github.com/ws-jiang/MetaDefense.

摘要: 本文介绍了MetaDefense，这是一种新型框架，用于防御大型语言模型（LLM）中基于微调的越狱攻击。我们观察到，尽管LLM能够区分嵌入空间中伪装的有害查询，但现有的防御机制未能推广到由看不见的攻击模板伪装的有害查询。基于这些见解，我们提出了一种两阶段防御方法：（i）一代前防御，在响应生成开始之前检测有害查询，以及（ii）中一代防御，在生成期间监视部分响应，以防止输出更多有害内容。我们的MetaDefense训练LLM使用专门的提示来预测查询和部分响应的危害性，从而能够提前终止潜在有害的交互。跨多个LLM架构（LLaMA-2- 7 B、Qwen-2.5- 3B-Direcct和LLaMA-3.2- 3B-Direcct）的广泛实验表明，Meta Defense的性能显着优于现有的防御机制，可以通过可见和不可见的攻击模板实现针对有害查询的稳健防御，同时在良性任务上保持有竞争力的性能。代码可在https://github.com/ws-jiang/MetaDefense上获取。



## **16. Effective and Stealthy One-Shot Jailbreaks on Deployed Mobile Vision-Language Agents**

在部署的移动视觉语言代理上进行有效且秘密的一次性越狱 cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07809v1) [paper-pdf](http://arxiv.org/pdf/2510.07809v1)

**Authors**: Renhua Ding, Xiao Yang, Zhengwei Fang, Jun Luo, Kun He, Jun Zhu

**Abstract**: Large vision-language models (LVLMs) enable autonomous mobile agents to operate smartphone user interfaces, yet vulnerabilities to UI-level attacks remain critically understudied. Existing research often depends on conspicuous UI overlays, elevated permissions, or impractical threat models, limiting stealth and real-world applicability. In this paper, we present a practical and stealthy one-shot jailbreak attack that leverages in-app prompt injections: malicious applications embed short prompts in UI text that remain inert during human interaction but are revealed when an agent drives the UI via ADB (Android Debug Bridge). Our framework comprises three crucial components: (1) low-privilege perception-chain targeting, which injects payloads into malicious apps as the agent's visual inputs; (2) stealthy user-invisible activation, a touch-based trigger that discriminates agent from human touches using physical touch attributes and exposes the payload only during agent operation; and (3) one-shot prompt efficacy, a heuristic-guided, character-level iterative-deepening search algorithm (HG-IDA*) that performs one-shot, keyword-level detoxification to evade on-device safety filters. We evaluate across multiple LVLM backends, including closed-source services and representative open-source models within three Android applications, and we observe high planning and execution hijack rates in single-shot scenarios (e.g., GPT-4o: 82.5% planning / 75.0% execution). These findings expose a fundamental security vulnerability in current mobile agents with immediate implications for autonomous smartphone operation.

摘要: 大型视觉语言模型（LVLM）使自主移动代理能够操作智能手机用户界面，但UI级攻击的漏洞仍然严重缺乏研究。现有的研究通常依赖于明显的UI覆盖、较高的权限或不切实际的威胁模型，从而限制了隐形性和现实世界的适用性。在本文中，我们提出了一种实用且隐蔽的一次性越狱攻击，该攻击利用应用程序内提示注入：恶意应用程序将短提示嵌入UI文本中，这些提示在人类交互期间保持惰性，但当代理通过ADB（Android Buttons Bridge）驱动UI时就会被泄露。我们的框架包括三个关键组件：（1）低特权感知链瞄准，将有效负载注入恶意应用程序作为代理的视觉输入;（2）隐形用户不可见激活，一种基于触摸的触发器，使用物理触摸属性将代理与人类触摸区分开来，并仅在代理操作期间暴露有效负载;和（3）一次性提示功效，一种启发式引导的字符级迭代深化搜索算法（HG-IDA*），可执行一次性关键字级解毒以逃避设备上的安全过滤器。我们对多个LVLM后台进行评估，包括三个Android应用程序中的开源服务和代表性开源模型，我们观察到单次场景中的规划和执行劫持率很高（例如，GPT-4 o：82.5%规划/ 75.0%执行）。这些发现暴露了当前移动代理中的一个根本安全漏洞，对自主智能手机操作产生了直接影响。



## **17. AEGIS : Automated Co-Evolutionary Framework for Guarding Prompt Injections Schema**

AEGIS：守卫提示注射模式的自动协同进化框架 cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2509.00088v2) [paper-pdf](http://arxiv.org/pdf/2509.00088v2)

**Authors**: Ting-Chun Liu, Ching-Yu Hsu, Kuan-Yi Lee, Chi-An Fu, Hung-yi Lee

**Abstract**: Prompt injection attacks pose a significant challenge to the safe deployment of Large Language Models (LLMs) in real-world applications. While prompt-based detection offers a lightweight and interpretable defense strategy, its effectiveness has been hindered by the need for manual prompt engineering. To address this issue, we propose AEGIS , an Automated co-Evolutionary framework for Guarding prompt Injections Schema. Both attack and defense prompts are iteratively optimized against each other using a gradient-like natural language prompt optimization technique. This framework enables both attackers and defenders to autonomously evolve via a Textual Gradient Optimization (TGO) module, leveraging feedback from an LLM-guided evaluation loop. We evaluate our system on a real-world assignment grading dataset of prompt injection attacks and demonstrate that our method consistently outperforms existing baselines, achieving superior robustness in both attack success and detection. Specifically, the attack success rate (ASR) reaches 1.0, representing an improvement of 0.26 over the baseline. For detection, the true positive rate (TPR) improves by 0.23 compared to the previous best work, reaching 0.84, and the true negative rate (TNR) remains comparable at 0.89. Ablation studies confirm the importance of co-evolution, gradient buffering, and multi-objective optimization. We also confirm that this framework is effective in different LLMs. Our results highlight the promise of adversarial training as a scalable and effective approach for guarding prompt injections.

摘要: 提示注入攻击对现实世界应用程序中大型语言模型（LLM）的安全部署构成了重大挑战。虽然基于预算的检测提供了一种轻量级且可解释的防御策略，但其有效性因需要手动提示工程而受到阻碍。为了解决这个问题，我们提出了AEGIS，这是Guarding提示注射模式的自动协同进化框架。攻击和防御提示都使用类似梯度的自然语言提示优化技术进行相互迭代优化。该框架使攻击者和防御者能够通过文本梯度优化（TGO）模块自主进化，利用来自LLM指导评估循环的反馈。我们在即时注入攻击的现实世界分配分级数据集上评估了我们的系统，并证明我们的方法始终优于现有基线，在攻击成功和检测方面都实现了卓越的鲁棒性。具体来说，攻击成功率（ASB）达到1.0，比基线提高0.26。在检测方面，真阳性率（TLR）与之前的最佳工作相比提高了0.23，达到0.84，真阴性率（TNR）保持在0.89相当。消融研究证实了共同进化、梯度缓冲和多目标优化的重要性。我们还确认该框架在不同的LLM中有效。我们的结果凸显了对抗训练作为一种可扩展且有效的预防及时注射方法的前景。



## **18. Rethinking Reasoning: A Survey on Reasoning-based Backdoors in LLMs**

重新思考推理：LLM中基于推理的后门调查 cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07697v1) [paper-pdf](http://arxiv.org/pdf/2510.07697v1)

**Authors**: Man Hu, Xinyi Wu, Zuofeng Suo, Jinbo Feng, Linghui Meng, Yanhao Jia, Anh Tuan Luu, Shuai Zhao

**Abstract**: With the rise of advanced reasoning capabilities, large language models (LLMs) are receiving increasing attention. However, although reasoning improves LLMs' performance on downstream tasks, it also introduces new security risks, as adversaries can exploit these capabilities to conduct backdoor attacks. Existing surveys on backdoor attacks and reasoning security offer comprehensive overviews but lack in-depth analysis of backdoor attacks and defenses targeting LLMs' reasoning abilities. In this paper, we take the first step toward providing a comprehensive review of reasoning-based backdoor attacks in LLMs by analyzing their underlying mechanisms, methodological frameworks, and unresolved challenges. Specifically, we introduce a new taxonomy that offers a unified perspective for summarizing existing approaches, categorizing reasoning-based backdoor attacks into associative, passive, and active. We also present defense strategies against such attacks and discuss current challenges alongside potential directions for future research. This work offers a novel perspective, paving the way for further exploration of secure and trustworthy LLM communities.

摘要: 随着高级推理能力的兴起，大型语言模型（LLM）越来越受到关注。然而，尽管推理提高了LLM在下游任务上的性能，但它也带来了新的安全风险，因为对手可以利用这些能力进行后门攻击。现有的关于后门攻击和推理安全性的调查提供了全面的概述，但缺乏对针对LLM推理能力的后门攻击和防御的深入分析。在本文中，我们通过分析LLM中基于推理的后门攻击的潜在机制、方法框架和未解决的挑战，迈出了对LLM中基于推理的后门攻击进行全面审查的第一步。具体来说，我们引入了一个新的分类法，提供了一个统一的角度来总结现有的方法，分类基于推理的后门攻击为关联，被动和主动。我们还提出了针对此类攻击的防御策略，并讨论了当前的挑战以及未来研究的潜在方向。这项工作提供了一个新的视角，为进一步探索安全和值得信赖的LLM社区铺平了道路。



## **19. LLM Unlearning Under the Microscope: A Full-Stack View on Methods and Metrics**

LLM显微镜下的学习：全栈视图的方法和技巧 cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07626v1) [paper-pdf](http://arxiv.org/pdf/2510.07626v1)

**Authors**: Chongyu Fan, Changsheng Wang, Yancheng Huang, Soumyadeep Pal, Sijia Liu

**Abstract**: Machine unlearning for large language models (LLMs) aims to remove undesired data, knowledge, and behaviors (e.g., for safety, privacy, or copyright) while preserving useful model capabilities. Despite rapid progress over the past two years, research in LLM unlearning remains fragmented, with limited clarity on what constitutes effective unlearning and how it should be rigorously evaluated. In this work, we present a principled taxonomy of twelve recent stateful unlearning methods, grouped into three methodological families: divergence-driven optimization, representation misalignment, and rejection-based targeted unlearning. Building on this taxonomy, we revisit the evaluation of unlearning effectiveness (UE), utility retention (UT), and robustness (Rob), focusing on the WMDP benchmark. Our analysis shows that current evaluations, dominated by multiple-choice question (MCQ) accuracy, offer only a narrow perspective, often overstating success while overlooking the model's actual generation behavior. To address this gap, we introduce open question-answering (Open-QA) metrics that better capture generative performance and reveal the inherent UE-UT tradeoff across method families. Furthermore, we demonstrate that robustness requires finer-grained analysis: for example, vulnerabilities differ substantially between in-domain relearning and out-of-domain fine-tuning, even though both fall under model-level attacks. Through this study, we hope to deliver a full-stack revisit of LLM unlearning and actionable guidance for designing and evaluating future methods.

摘要: 大型语言模型（LLM）的机器去学习旨在删除不需要的数据、知识和行为（例如，为了安全、隐私或版权），同时保留有用的模型功能。尽管过去两年取得了迅速的进展，但LLM忘记学习的研究仍然支离破碎，对于什么构成有效的忘记学习以及如何严格评估它的清晰度有限。在这项工作中，我们对最近的十二种有状态的去学习方法提出了原则性的分类，分为三个方法族：分歧驱动的优化、表示失准和基于拒绝的有针对性的去学习。在此分类法的基础上，我们重新审视了对取消学习有效性（UE）、效用保留（UT）和稳健性（Rob）的评估，重点关注WMDP基准。我们的分析表明，当前的评估以多项选择题（MCQ）准确性为主，仅提供了狭隘的视角，经常夸大成功，而忽视了模型的实际生成行为。为了解决这一差距，我们引入了开放式问答（Open-QA）指标，可以更好地捕捉生成性能并揭示方法系列之间固有的UE-UT权衡。此外，我们证明稳健性需要更细粒度的分析：例如，域内重新学习和域外微调之间的漏洞存在很大差异，尽管两者都受到模型级攻击。通过这项研究，我们希望对LLM的遗忘进行全面回顾，并为设计和评估未来的方法提供可操作的指导。



## **20. $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks**

$\texttit {Agents Under Siege}$：通过优化的即时攻击破解实用多Agent LLM系统 cs.MA

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2504.00218v2) [paper-pdf](http://arxiv.org/pdf/2504.00218v2)

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Fleming, Tianlong Chen

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.

摘要: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.



## **21. PEAR: Planner-Executor Agent Robustness Benchmark**

PEAR：规划者-执行者代理稳健性基准 cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07505v1) [paper-pdf](http://arxiv.org/pdf/2510.07505v1)

**Authors**: Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.

摘要: 基于大型语言模型（LLM）的多智能体系统（MAS）已成为处理跨不同领域复杂、多步骤任务的强大范式。然而，尽管MAS的能力令人印象深刻，但仍然容易受到对抗操纵。现有的研究通常会检查孤立的攻击表面或特定场景，从而缺乏对MAS漏洞的全面了解。为了弥合这一差距，我们引入了PEAR，这是一个用于系统评估规划者-执行者MAS的实用性和脆弱性的基准。虽然兼容各种MAS体系结构，我们的基准集中在规划者-执行器结构，这是一个实用的和广泛采用的设计。通过大量的实验，我们发现：（1）弱规划器比弱执行器更严重地降低了清洁任务的整体性能;（2）虽然规划器的内存模块是必不可少的，但执行器的内存模块并不影响清洁任务的性能;（3）任务性能和鲁棒性之间存在权衡;以及（4）针对计划者的攻击在误导系统方面特别有效。这些发现提供了可操作的见解，提高MAS的鲁棒性，并奠定了基础，在多智能体设置的原则性防御。



## **22. L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint)**

L2 M-AID：通过融合大型语言模型的语义推理与多智能体强化学习来自主网络物理防御（预印本） cs.AI

This preprint was submitted to IEEE TrustCom 2025. The accepted  version will be published under copyright 2025 IEEE

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07363v1) [paper-pdf](http://arxiv.org/pdf/2510.07363v1)

**Authors**: Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Jun Wang, Yan Li, Chang Liu

**Abstract**: The increasing integration of Industrial IoT (IIoT) exposes critical cyber-physical systems to sophisticated, multi-stage attacks that elude traditional defenses lacking contextual awareness. This paper introduces L2M-AID, a novel framework for Autonomous Industrial Defense using LLM-empowered, Multi-agent reinforcement learning. L2M-AID orchestrates a team of collaborative agents, each driven by a Large Language Model (LLM), to achieve adaptive and resilient security. The core innovation lies in the deep fusion of two AI paradigms: we leverage an LLM as a semantic bridge to translate vast, unstructured telemetry into a rich, contextual state representation, enabling agents to reason about adversary intent rather than merely matching patterns. This semantically-aware state empowers a Multi-Agent Reinforcement Learning (MARL) algorithm, MAPPO, to learn complex cooperative strategies. The MARL reward function is uniquely engineered to balance security objectives (threat neutralization) with operational imperatives, explicitly penalizing actions that disrupt physical process stability. To validate our approach, we conduct extensive experiments on the benchmark SWaT dataset and a novel synthetic dataset generated based on the MITRE ATT&CK for ICS framework. Results demonstrate that L2M-AID significantly outperforms traditional IDS, deep learning anomaly detectors, and single-agent RL baselines across key metrics, achieving a 97.2% detection rate while reducing false positives by over 80% and improving response times by a factor of four. Crucially, it demonstrates superior performance in maintaining physical process stability, presenting a robust new paradigm for securing critical national infrastructure.

摘要: 工业物联网（IIoT）的日益集成使关键的网络物理系统面临复杂的多阶段攻击，这些攻击无法逃避缺乏上下文感知的传统防御。本文介绍了L2 M-AID，这是一种新型的自主工业防御框架，使用LLM授权的多智能体强化学习。L2 M-AID组织了一个协作代理团队，每个代理都由大型语言模型（LLM）驱动，以实现自适应和弹性的安全性。核心创新在于两种人工智能范式的深度融合：我们利用LLM作为语义桥梁，将庞大的非结构化遥感数据转化为丰富的上下文状态表示，使代理能够推理对手意图，而不仅仅是匹配模式。这种语义感知状态使多智能体强化学习（MARL）算法MAPPO能够学习复杂的合作策略。MARL奖励功能经过独特设计，旨在平衡安全目标（威胁消除）与运营必要性，明确惩罚破坏物理过程稳定性的行为。为了验证我们的方法，我们对基准SWaT数据集和基于MITRE ATA & CK for ICS框架生成的新型合成数据集进行了广泛的实验。结果表明，L2 M-AID在关键指标上的表现显着优于传统IDS、深度学习异常检测器和单代理RL基线，实现了97.2%的检测率，同时将误报率降低了80%以上，并将响应时间提高了四倍。至关重要的是，它在维持物理过程稳定性方面表现出色，为保护关键国家基础设施提供了强大的新范式。



## **23. Red-Bandit: Test-Time Adaptation for LLM Red-Teaming via Bandit-Guided LoRA Experts**

Red-Bandit：通过Bandit-Guided LoRA专家进行LLM Red-Teaming的测试时适应 cs.CL

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07239v1) [paper-pdf](http://arxiv.org/pdf/2510.07239v1)

**Authors**: Christos Ziakas, Nicholas Loo, Nishita Jain, Alessandra Russo

**Abstract**: Automated red-teaming has emerged as a scalable approach for auditing Large Language Models (LLMs) prior to deployment, yet existing approaches lack mechanisms to efficiently adapt to model-specific vulnerabilities at inference. We introduce Red-Bandit, a red-teaming framework that adapts online to identify and exploit model failure modes under distinct attack styles (e.g., manipulation, slang). Red-Bandit post-trains a set of parameter-efficient LoRA experts, each specialized for a particular attack style, using reinforcement learning that rewards the generation of unsafe prompts via a rule-based safety model. At inference, a multi-armed bandit policy dynamically selects among these attack-style experts based on the target model's response safety, balancing exploration and exploitation. Red-Bandit achieves state-of-the-art results on AdvBench under sufficient exploration (ASR@10), while producing more human-readable prompts (lower perplexity). Moreover, Red-Bandit's bandit policy serves as a diagnostic tool for uncovering model-specific vulnerabilities by indicating which attack styles most effectively elicit unsafe behaviors.

摘要: 自动红色团队已成为一种在部署之前审计大型语言模型（LLM）的可扩展方法，但现有方法缺乏有效适应推理时特定于模型的漏洞的机制。我们引入了Red-Bandit，这是一个红色团队框架，可以在线调整以识别和利用不同攻击风格下的模型故障模式（例如，操纵，俚语）。Red-Bandit对一组参数高效的LoRA专家进行后期培训，每个专家都专门针对特定的攻击风格，使用强化学习通过基于规则的安全模型奖励不安全提示的生成。推断，多武装强盗政策根据目标模型的响应安全性在这些攻击风格的专家中动态选择，平衡探索和利用。Red-Bandit在充分探索下（ASR@10）在AdvBench上实现了最先进的结果，同时生成更多人类可读的提示（更低的困惑度）。此外，Red-Bandit的强盗策略还充当诊断工具，通过指示哪些攻击风格最有效地引发不安全行为来发现特定于模型的漏洞。



## **24. Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples**

对LLM的中毒攻击需要几乎恒定数量的毒物样本 cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07192v1) [paper-pdf](http://arxiv.org/pdf/2510.07192v1)

**Authors**: Alexandra Souly, Javier Rando, Ed Chapman, Xander Davies, Burak Hasircioglu, Ezzeldin Shereen, Carlos Mougan, Vasilios Mavroudis, Erik Jones, Chris Hicks, Nicholas Carlini, Yarin Gal, Robert Kirk

**Abstract**: Poisoning attacks can compromise the safety of large language models (LLMs) by injecting malicious documents into their training data. Existing work has studied pretraining poisoning assuming adversaries control a percentage of the training corpus. However, for large models, even small percentages translate to impractically large amounts of data. This work demonstrates for the first time that poisoning attacks instead require a near-constant number of documents regardless of dataset size. We conduct the largest pretraining poisoning experiments to date, pretraining models from 600M to 13B parameters on chinchilla-optimal datasets (6B to 260B tokens). We find that 250 poisoned documents similarly compromise models across all model and dataset sizes, despite the largest models training on more than 20 times more clean data. We also run smaller-scale experiments to ablate factors that could influence attack success, including broader ratios of poisoned to clean data and non-random distributions of poisoned samples. Finally, we demonstrate the same dynamics for poisoning during fine-tuning. Altogether, our results suggest that injecting backdoors through data poisoning may be easier for large models than previously believed as the number of poisons required does not scale up with model size, highlighting the need for more research on defences to mitigate this risk in future models.

摘要: 中毒攻击可能会通过将恶意文档注入大型语言模型（LLM）的训练数据中来危及大型语言模型（LLM）的安全性。现有的工作已经研究了训练前中毒，假设对手控制了一定比例的训练素材。然而，对于大型模型来说，即使是很小的百分比也会转化为不切实际的大量数据。这项工作首次证明，无论数据集大小如何，中毒攻击都需要几乎恒定数量的文档。我们进行了迄今为止最大的预训练中毒实验，在龙猫最佳数据集（6 B至260 B代币）上预训练600 M至13 B参数的模型。我们发现，尽管最大的模型在干净数据上训练了20倍以上的数据，但250个有毒文档同样会损害所有模型和数据集大小的模型。我们还进行了较小规模的实验，以消除可能影响攻击成功的因素，包括更广泛的中毒数据与干净数据的比例以及中毒样本的非随机分布。最后，我们演示了微调期间中毒的相同动态。总而言之，我们的结果表明，对于大型模型来说，通过数据中毒注入后门可能比之前认为的更容易，因为所需的毒药数量不会随着模型大小而增加，这凸显了需要对防御进行更多研究，以减轻未来模型中的这种风险。



## **25. RedTWIZ: Diverse LLM Red Teaming via Adaptive Attack Planning**

RedTWIZ：通过自适应攻击规划实现多元化LLM红色团队 cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06994v1) [paper-pdf](http://arxiv.org/pdf/2510.06994v1)

**Authors**: Artur Horal, Daniel Pina, Henrique Paz, Iago Paulo, João Soares, Rafael Ferreira, Diogo Tavares, Diogo Glória-Silva, João Magalhães, David Semedo

**Abstract**: This paper presents the vision, scientific contributions, and technical details of RedTWIZ: an adaptive and diverse multi-turn red teaming framework, to audit the robustness of Large Language Models (LLMs) in AI-assisted software development. Our work is driven by three major research streams: (1) robust and systematic assessment of LLM conversational jailbreaks; (2) a diverse generative multi-turn attack suite, supporting compositional, realistic and goal-oriented jailbreak conversational strategies; and (3) a hierarchical attack planner, which adaptively plans, serializes, and triggers attacks tailored to specific LLM's vulnerabilities. Together, these contributions form a unified framework -- combining assessment, attack generation, and strategic planning -- to comprehensively evaluate and expose weaknesses in LLMs' robustness. Extensive evaluation is conducted to systematically assess and analyze the performance of the overall system and each component. Experimental results demonstrate that our multi-turn adversarial attack strategies can successfully lead state-of-the-art LLMs to produce unsafe generations, highlighting the pressing need for more research into enhancing LLM's robustness.

摘要: 本文介绍了RedTWIZ的愿景、科学贡献和技术细节：一个自适应且多样化的多回合红色团队框架，用于审核大型语言模型（LLM）在人工智能辅助软件开发中的稳健性。我们的工作由三个主要研究流推动：（1）对LLM对话越狱的稳健和系统性评估;（2）多元化的生成式多回合攻击套件，支持组合性、现实性和面向目标的越狱对话策略;（3）分层攻击规划器，它自适应地规划、序列化和触发针对特定LLM漏洞的攻击。这些贡献共同构成了一个统一的框架--结合了评估、攻击生成和战略规划--以全面评估和揭露LLM稳健性的弱点。进行广泛的评估，以系统地评估和分析整个系统和每个组件的性能。实验结果表明，我们的多回合对抗攻击策略可以成功导致最先进的LLM产生不安全的世代，凸显了对增强LLM稳健性进行更多研究的迫切需要。



## **26. VelLMes: A high-interaction AI-based deception framework**

VelLMes：一个高交互性的基于人工智能的欺骗框架 cs.CR

9 pages. 9 figures. 1 table. This is a preprint of a paper that was  presented at the Active Defense and Deception Workshop colocated with IEEE  EuroS&P 2025 conference

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06975v1) [paper-pdf](http://arxiv.org/pdf/2510.06975v1)

**Authors**: Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia

**Abstract**: There are very few SotA deception systems based on Large Language Models. The existing ones are limited only to simulating one type of service, mainly SSH shells. These systems - but also the deception technologies not based on LLMs - lack an extensive evaluation that includes human attackers. Generative AI has recently become a valuable asset for cybersecurity researchers and practitioners, and the field of cyber-deception is no exception. Researchers have demonstrated how LLMs can be leveraged to create realistic-looking honeytokens, fake users, and even simulated systems that can be used as honeypots. This paper presents an AI-based deception framework called VelLMes, which can simulate multiple protocols and services such as SSH Linux shell, MySQL, POP3, and HTTP. All of these can be deployed and used as honeypots, thus VelLMes offers a variety of choices for deception design based on the users' needs. VelLMes is designed to be attacked by humans, so interactivity and realism are key for its performance. We evaluate the generative capabilities and the deception capabilities. Generative capabilities were evaluated using unit tests for LLMs. The results of the unit tests show that, with careful prompting, LLMs can produce realistic-looking responses, with some LLMs having a 100% passing rate. In the case of the SSH Linux shell, we evaluated deception capabilities with 89 human attackers. The results showed that about 30% of the attackers thought that they were interacting with a real system when they were assigned an LLM-based honeypot. Lastly, we deployed 10 instances of the SSH Linux shell honeypot on the Internet to capture real-life attacks. Analysis of these attacks showed us that LLM honeypots simulating Linux shells can perform well against unstructured and unexpected attacks on the Internet, responding correctly to most of the issued commands.

摘要: 基于大型语言模型的SotA欺骗系统很少。现有的服务仅限于模拟一种类型的服务，主要是SSH shell。这些系统--以及不基于LLM的欺骗技术--缺乏包括人类攻击者在内的广泛评估。生成性人工智能最近已成为网络安全研究人员和从业者的宝贵资产，网络欺骗领域也不例外。研究人员已经演示了如何利用LLM来创建外观逼真的蜜罐、虚假用户，甚至可用作蜜罐的模拟系统。本文提出了一个基于人工智能的欺骗框架VelLMes，它可以模拟多种协议和服务，例如SSH Linux shell、SQL、POP3和HTTP。所有这些都可以作为蜜罐部署和使用，因此VelLMes根据用户的需求提供了多种欺骗设计选择。VelLMes旨在被人类攻击，因此交互性和真实感是其性能的关键。我们评估生成能力和欺骗能力。使用LLM的单元测试评估生成能力。单元测试的结果表明，在仔细提示下，LLM可以产生看起来真实的响应，部分LLM的通过率达到100%。以SSH Linux shell为例，我们评估了89名人类攻击者的欺骗能力。结果显示，当被分配基于LLM的蜜罐时，大约30%的攻击者认为他们正在与真实的系统交互。最后，我们在互联网上部署了10个SSH Linux shell蜜罐实例，以捕获现实生活中的攻击。对这些攻击的分析表明，模拟Linux shell的LLM蜜罐可以很好地应对互联网上的非结构化和意外攻击，并正确响应大多数发布的命令。



## **27. Exposing Citation Vulnerabilities in Generative Engines**

暴露生成引擎中的引用漏洞 cs.CR

12 pages, under-reviewing at a conference

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06823v1) [paper-pdf](http://arxiv.org/pdf/2510.06823v1)

**Authors**: Riku Mochizuki, Shusuke Komatsu, Souta Noguchi, Kazuto Ataka

**Abstract**: We analyze answers generated by generative engines (GEs) from the perspectives of citation publishers and the content-injection barrier, defined as the difficulty for attackers to manipulate answers to user prompts by placing malicious content on the web. GEs integrate two functions: web search and answer generation that cites web pages using large language models. Because anyone can publish information on the web, GEs are vulnerable to poisoning attacks. Existing studies of citation evaluation focus on how faithfully answer content reflects cited sources, leaving unexamined which web sources should be selected as citations to defend against poisoning attacks. To fill this gap, we introduce evaluation criteria that assess poisoning threats using the citation information contained in answers. Our criteria classify the publisher attributes of citations to estimate the content-injection barrier thereby revealing the threat of poisoning attacks in current GEs. We conduct experiments in political domains in Japan and the United States (U.S.) using our criteria and show that citations from official party websites (primary sources) are approximately \(25\%\)--\(45\%\) in the U.S. and \(60\%\)--\(65\%\) in Japan, indicating that U.S. political answers are at higher risk of poisoning attacks. We also find that sources with low content-injection barriers are frequently cited yet are poorly reflected in answer content. To mitigate this threat, we discuss how publishers of primary sources can increase exposure of their web content in answers and show that well-known techniques are limited by language differences.

摘要: 我们从引文发布者和内容注入障碍的角度分析生成引擎（GE）生成的答案，内容注入障碍被定义为攻击者通过在网络上放置恶意内容来操纵用户提示答案的难度。GE集成了两项功能：网络搜索和使用大型语言模型引用网页的答案生成。由于任何人都可以在网络上发布信息，GE很容易受到中毒攻击。现有的引文评估研究重点关注回答内容如何忠实地反映引用的来源，而没有审查应该选择哪些网络来源作为引文以防御中毒攻击。为了填补这一空白，我们引入评估标准，评估中毒的威胁使用的引文信息中包含的答案。我们的标准分类的出版商属性的引文，以估计内容注入障碍，从而揭示了中毒攻击的威胁，在当前的GE。我们在日本和美国进行政治领域的实验。使用我们的标准，并显示来自官方政党网站（主要来源）的引用在美国约为\（25\%\）-\（45\%\），在日本约为\（60\%\）-\（65\%\），这表明美国的政治答案受到毒害攻击的风险更高。我们还发现，内容注入障碍低的来源经常被引用，但在答案内容中反映得很差。为了减轻这种威胁，我们讨论了主要来源的出版商如何在答案中增加其网络内容的暴露度，并表明众所周知的技术受到语言差异的限制。



## **28. Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness**

获取财富或死亡缩放：盈利交易推理计算以实现稳健性 cs.LG

17 pages

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06790v1) [paper-pdf](http://arxiv.org/pdf/2510.06790v1)

**Authors**: Tavish McDonald, Bo Lei, Stanislav Fort, Bhavya Kailkhura, Brian Bartoldson

**Abstract**: Models are susceptible to adversarially out-of-distribution (OOD) data despite large training-compute investments into their robustification. Zaremba et al. (2025) make progress on this problem at test time, showing LLM reasoning improves satisfaction of model specifications designed to thwart attacks, resulting in a correlation between reasoning effort and robustness to jailbreaks. However, this benefit of test compute fades when attackers are given access to gradients or multimodal inputs. We address this gap, clarifying that inference-compute offers benefits even in such cases. Our approach argues that compositional generalization, through which OOD data is understandable via its in-distribution (ID) components, enables adherence to defensive specifications on adversarially OOD inputs. Namely, we posit the Robustness from Inference Compute Hypothesis (RICH): inference-compute defenses profit as the model's training data better reflects the attacked data's components. We empirically support this hypothesis across vision language model and attack types, finding robustness gains from test-time compute if specification following on OOD data is unlocked by compositional generalization, while RL finetuning and protracted reasoning are not critical. For example, increasing emphasis on defensive specifications via prompting lowers the success rate of gradient-based multimodal attacks on VLMs robustified by adversarial pretraining, but this same intervention provides no such benefit to not-robustified models. This correlation of inference-compute's robustness benefit with base model robustness is the rich-get-richer dynamic of the RICH: attacked data components are more ID for robustified models, aiding compositional generalization to OOD data. Accordingly, we advise layering train-time and test-time defenses to obtain their synergistic benefit.

摘要: 尽管模型的鲁棒性投入了大量的训练计算投资，但它们仍然容易受到不利的分布外（OOD）数据的影响。Zaremba等人（2025）在测试时在这个问题上取得了进展，表明LLM推理提高了旨在阻止攻击的模型规范的满意度，从而导致推理工作量和越狱稳健性之间的相关性。然而，当攻击者能够访问梯度或多模式输入时，测试计算的这种好处就会消失。我们解决了这一差距，澄清了即使在这种情况下，推理计算也能带来好处。我们的方法认为，组合概括（OOD数据可以通过其内分布（ID）组件来理解）使得能够遵守针对敌对OOD输入的防御规范。也就是说，我们从推理计算假设（RICH）中验证了鲁棒性：由于模型的训练数据更好地反映了受攻击数据的成分，推理计算防御会获利。我们在视觉语言模型和攻击类型中从经验上支持了这一假设，如果OOD数据上的规范通过组合概括解锁，则可以从测试时计算中找到鲁棒性收益，而RL微调和持久推理并不关键。例如，通过提示来增加对防御规范的强调会降低对由对抗性预训练稳健的VLM的基于梯度的多模式攻击的成功率，但同样的干预并没有为非稳健的模型提供这样的好处。推理计算的鲁棒性与基础模型鲁棒性的这种相关性是RICH的丰富-越来越丰富的动态：受攻击的数据组件对于鲁棒模型来说更具ID，有助于组合泛化到OOD数据。因此，我们建议分层训练时和测试时的防御，以获得其协同效益。



## **29. Benchmarking Gaslighting Negation Attacks Against Multimodal Large Language Models**

针对多模式大型语言模型的Gaslighting否定攻击基准 cs.CL

Project website:  https://yxg1005.github.io/GaslightingNegationAttacks/

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2501.19017v4) [paper-pdf](http://arxiv.org/pdf/2501.19017v4)

**Authors**: Bin Zhu, Yinxuan Gui, Huiyan Qi, Jingjing Chen, Chong-Wah Ngo, Ee-Peng Lim

**Abstract**: Multimodal Large Language Models (MLLMs) have exhibited remarkable advancements in integrating different modalities, excelling in complex understanding and generation tasks. Despite their success, MLLMs remain vulnerable to conversational adversarial inputs. In this paper, we systematically study gaslighting negation attacks: a phenomenon where models, despite initially providing correct answers, are persuaded by user-provided negations to reverse their outputs, often fabricating justifications. We conduct extensive evaluations of state-of-the-art MLLMs across diverse benchmarks and observe substantial performance drops when negation is introduced. Notably, we introduce the first benchmark GaslightingBench, specifically designed to evaluate the vulnerability of MLLMs to negation arguments. GaslightingBench consists of multiple-choice questions curated from existing datasets, along with generated negation prompts across 20 diverse categories. Throughout extensive evaluation, we find that proprietary models such as Gemini-1.5-flash and GPT-4o demonstrate better resilience compared to open-source counterparts like Qwen2-VL and LLaVA, though even advanced reasoning-oriented models like Gemini-2.5-Pro remain susceptible. Our category-level analysis further shows that subjective or socially nuanced domains (e.g., Social Relation, Image Emotion) are especially fragile, while more objective domains (e.g., Geography) exhibit relatively smaller but still notable drops. Overall, all evaluated MLLMs struggle to maintain logical consistency under gaslighting negation attack. These findings highlight a fundamental robustness gap and provide insights for developing more reliable and trustworthy multimodal AI systems. Project website: https://yxg1005.github.io/GaslightingNegationAttacks/.

摘要: 多模式大型语言模型（MLLM）在集成不同模式方面表现出了显着的进步，在复杂的理解和生成任务中表现出色。尽管取得了成功，MLLM仍然容易受到对话对抗输入的影响。在本文中，我们系统地研究了煤气灯否定攻击：这是一种现象，模型尽管最初提供了正确的答案，但被用户提供的否定说服来扭转其输出，通常编造理由。我们对不同基准的最先进的MLLM进行了广泛评估，并观察到当引入否定时性能会大幅下降。值得注意的是，我们引入了第一个基准GaslightingBench，专门用于评估MLLM对否定论点的脆弱性。GaslightingBench由根据现有数据集精心设计的多项选择题以及生成的跨越20个不同类别的否定提示组成。在广泛的评估中，我们发现Gemini-1.5-Flash和GPT-4 o等专有模型与Qwen 2-BL和LLaVA等开源模型相比表现出更好的弹性，尽管即使是像Gemini-2.5-Pro这样的高级推理导向模型仍然容易受到影响。我们的类别级分析进一步表明，主观或社会细微差别领域（例如，社会关系，形象情感）尤其脆弱，而更客观的领域（例如，地理）显示相对较小，但仍然显着下降。总体而言，所有评估的MLLM努力保持逻辑一致性下gaslighting否定攻击。这些发现突出了一个基本的鲁棒性差距，并为开发更可靠和值得信赖的多模态AI系统提供了见解。项目网站：https://yxg1005.github.io/GaslightingNegationAttacks/。



## **30. Enhancing GraphQL Security by Detecting Malicious Queries Using Large Language Models, Sentence Transformers, and Convolutional Neural Networks**

通过使用大型语言模型、句子转换器和卷积神经网络检测恶意语法来增强GraphQL安全性 cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2508.11711v2) [paper-pdf](http://arxiv.org/pdf/2508.11711v2)

**Authors**: Irash Perera, Hiranya Abeyrathne, Sanjeewa Malalgoda, Arshardh Ifthikar

**Abstract**: GraphQL's flexibility, while beneficial for efficient data fetching, introduces unique security vulnerabilities that traditional API security mechanisms often fail to address. Malicious GraphQL queries can exploit the language's dynamic nature, leading to denial-of-service attacks, data exfiltration through injection, and other exploits. Existing solutions, such as static analysis, rate limiting, and general-purpose Web Application Firewalls, offer limited protection against sophisticated, context-aware attacks. This paper presents a novel, AI-driven approach for real-time detection of malicious GraphQL queries. Our method combines static analysis with machine learning techniques, including Large Language Models (LLMs) for dynamic schema-based configuration, Sentence Transformers (SBERT and Doc2Vec) for contextual embedding of query payloads, and Convolutional Neural Networks (CNNs), Random Forests, and Multilayer Perceptrons for classification. We detail the system architecture, implementation strategies optimized for production environments (including ONNX Runtime optimization and parallel processing), and evaluate the performance of our detection models and the overall system under load. Results demonstrate high accuracy in detecting various threats, including SQL injection, OS command injection, and XSS exploits, alongside effective mitigation of DoS and SSRF attempts. This research contributes a robust and adaptable solution for enhancing GraphQL API security.

摘要: GraphQL的灵活性虽然有利于高效的数据获取，但也引入了传统API安全机制通常无法解决的独特安全漏洞。恶意的GraphQL查询可以利用该语言的动态性质，导致拒绝服务攻击、通过注入的数据泄露和其他利用。现有的解决方案（例如静态分析、速率限制和通用Web应用程序防火墙）只能针对复杂的上下文感知攻击提供有限的保护。本文提出了一种新颖的人工智能驱动方法，用于实时检测恶意GraphQL查询。我们的方法将静态分析与机器学习技术相结合，包括用于基于动态模式的配置的大型语言模型（LLM）、用于查询有效负载的上下文嵌入的句子转换器（SBERT和Doc 2Vec），以及用于分类的卷积神经网络（CNN）、随机森林和多层感知器。我们详细介绍了针对生产环境优化的系统架构、实施策略（包括ONNX收件箱优化和并行处理），并评估我们的检测模型和整个系统负载下的性能。结果表明，在检测各种威胁（包括SQL注入、OS命令注入和XSS漏洞利用）方面具有高准确性，并且有效缓解了DPS和SSRF尝试。这项研究为增强GraphQL API安全性提供了一个强大且适应性强的解决方案。



## **31. Membership Inference Attacks on LLM-based Recommender Systems**

对基于LLM的推荐系统的成员推断攻击 cs.IR

this paper is under review

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2508.18665v3) [paper-pdf](http://arxiv.org/pdf/2508.18665v3)

**Authors**: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen

**Abstract**: Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.

摘要: 基于大型语言模型（LLM）的推荐系统（RecSys）可以灵活地调整推荐系统以适应不同的领域。它利用上下文学习（ICL），即提示来定制推荐功能，其中包括敏感的历史用户特定项目交互，例如，隐性反馈，例如点击的项目或明确的产品评论。这样的私人信息可能暴露于新的隐私攻击。然而，对于这个重要问题还没有进行任何研究。我们设计了四个成员推理攻击（MIA），旨在揭示受害者的历史互动是否已被系统提示使用。它们是直接询问，幻觉，相似性和中毒攻击，每一种都利用了LLM或RecSys的独特功能。我们在三个用于开发ICL-LLM RecSys的LLM和两个著名的RecSys基准数据集上仔细评估了它们。结果证实，LLM RecSys上的MIA威胁是现实的：直接询问和中毒攻击显示出显着较高的攻击优势。我们还分析了影响这些攻击的因素，例如系统提示中的射击次数以及受害者在射击中的位置。



## **32. AutoDAN-Reasoning: Enhancing Strategies Exploration based Jailbreak Attacks with Test-Time Scaling**

AutoDAN推理：通过测试时间缩放增强基于策略探索的越狱攻击 cs.CR

Technical report. Code is available at  https://github.com/SaFoLab-WISC/AutoDAN-Reasoning

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.05379v2) [paper-pdf](http://arxiv.org/pdf/2510.05379v2)

**Authors**: Xiaogeng Liu, Chaowei Xiao

**Abstract**: Recent advancements in jailbreaking large language models (LLMs), such as AutoDAN-Turbo, have demonstrated the power of automated strategy discovery. AutoDAN-Turbo employs a lifelong learning agent to build a rich library of attack strategies from scratch. While highly effective, its test-time generation process involves sampling a strategy and generating a single corresponding attack prompt, which may not fully exploit the potential of the learned strategy library. In this paper, we propose to further improve the attack performance of AutoDAN-Turbo through test-time scaling. We introduce two distinct scaling methods: Best-of-N and Beam Search. The Best-of-N method generates N candidate attack prompts from a sampled strategy and selects the most effective one based on a scorer model. The Beam Search method conducts a more exhaustive search by exploring combinations of strategies from the library to discover more potent and synergistic attack vectors. According to the experiments, the proposed methods significantly boost performance, with Beam Search increasing the attack success rate by up to 15.6 percentage points on Llama-3.1-70B-Instruct and achieving a nearly 60% relative improvement against the highly robust GPT-o4-mini compared to the vanilla method.

摘要: 越狱大型语言模型（LLM）（例如AutoDAN-Turbo）的最新进展证明了自动策略发现的力量。AutoDAN-Turbo采用终身学习代理从头开始构建丰富的攻击策略库。虽然非常有效，但其测试时生成过程涉及对策略进行采样并生成单个相应的攻击提示，这可能无法充分利用所学到的策略库的潜力。本文建议通过测试时间扩展进一步提高AutoDAN-Turbo的攻击性能。我们介绍了两种不同的缩放方法：Best-of-N和Beam Search。N中最佳方法从抽样策略中产生N个候选攻击提示，并基于评分器模型选择最有效的一个。Beam Search方法通过探索库中的策略组合来进行更详尽的搜索，以发现更强大和协同的攻击向量。根据实验，与香草方法相比，提出的方法显着提高了性能，Beam Search将Llama-3.1- 70 B-Direct上的攻击成功率提高了15.6个百分点，并且相对于高度稳健的GPT-o 4-mini实现了近60%的相对改进。



## **33. Towards the Worst-case Robustness of Large Language Models**

走向大型语言模型的最坏情况稳健性 cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2501.19040v4) [paper-pdf](http://arxiv.org/pdf/2501.19040v4)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of large language models to adversarial attacks, where adversaries craft specific input sequences to induce harmful, violent, private, or incorrect outputs. In this work, we study their worst-case robustness, i.e., whether an adversarial example exists that leads to such undesirable outputs. We upper bound the worst-case robustness using stronger white-box attacks, indicating that most current deterministic defenses achieve nearly 0\% worst-case robustness. We propose a general tight lower bound for randomized smoothing using fractional knapsack solvers or 0-1 knapsack solvers, and using them to bound the worst-case robustness of all stochastic defenses. Based on these solvers, we provide theoretical lower bounds for several previous empirical defenses. For example, we certify the robustness of a specific case, smoothing using a uniform kernel, against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.

摘要: 最近的研究揭示了大型语言模型容易受到对抗攻击，对手会精心设计特定的输入序列来引发有害、暴力、私密或错误的输出。在这项工作中，我们研究了它们的最坏情况稳健性，即是否存在导致此类不良结果的对抗性例子。我们使用更强的白盒攻击来对最坏情况的稳健性进行上限，这表明当前大多数确定性防御实现了近0%的最坏情况的稳健性。我们提出了使用分数背包求解器或0-1背包求解器的随机平滑的一般紧下界，并使用它们来限制所有随机防御的最坏情况稳健性。基于这些求解器，我们为之前的几个经验防御提供了理论下限。例如，我们证明了特定情况的稳健性，使用统一核进行平滑，针对\texttit {任何可能的攻击}，平均$\ell_0 $扰动为2.02或平均后缀长度为6.41。



## **34. Code Agent can be an End-to-end System Hacker: Benchmarking Real-world Threats of Computer-use Agent**

代码代理可以成为端到端系统黑客：对计算机使用代理的现实世界威胁进行基准测试 cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06607v1) [paper-pdf](http://arxiv.org/pdf/2510.06607v1)

**Authors**: Weidi Luo, Qiming Zhang, Tianyu Lu, Xiaogeng Liu, Bin Hu, Hung-Chun Chiu, Siyuan Ma, Yizhe Zhang, Xusheng Xiao, Yinzhi Cao, Zhen Xiang, Chaowei Xiao

**Abstract**: Computer-use agent (CUA) frameworks, powered by large language models (LLMs) or multimodal LLMs (MLLMs), are rapidly maturing as assistants that can perceive context, reason, and act directly within software environments. Among their most critical applications is operating system (OS) control. As CUAs in the OS domain become increasingly embedded in daily operations, it is imperative to examine their real-world security implications, specifically whether CUAs can be misused to perform realistic, security-relevant attacks. Existing works exhibit four major limitations: Missing attacker-knowledge model on tactics, techniques, and procedures (TTP), Incomplete coverage for end-to-end kill chains, unrealistic environment without multi-host and encrypted user credentials, and unreliable judgment dependent on LLM-as-a-Judge. To address these gaps, we propose AdvCUA, the first benchmark aligned with real-world TTPs in MITRE ATT&CK Enterprise Matrix, which comprises 140 tasks, including 40 direct malicious tasks, 74 TTP-based malicious tasks, and 26 end-to-end kill chains, systematically evaluates CUAs under a realistic enterprise OS security threat in a multi-host environment sandbox by hard-coded evaluation. We evaluate the existing five mainstream CUAs, including ReAct, AutoGPT, Gemini CLI, Cursor CLI, and Cursor IDE based on 8 foundation LLMs. The results demonstrate that current frontier CUAs do not adequately cover OS security-centric threats. These capabilities of CUAs reduce dependence on custom malware and deep domain expertise, enabling even inexperienced attackers to mount complex enterprise intrusions, which raises social concern about the responsibility and security of CUAs.

摘要: 由大型语言模型（LLM）或多模式LLM（MLLM）支持的计算机使用代理（CUA）框架正在迅速成熟，成为可以感知上下文、推理和直接在软件环境中采取行动的助手。它们最关键的应用程序之一是操作系统（OS）控制。随着操作系统领域的CUA越来越嵌入到日常操作中，必须检查其现实世界的安全影响，特别是是否可以滥用CUA来执行现实的、安全相关的攻击。现有作品表现出四个主要局限性：缺乏关于战术、技术和程序（TTP）的攻击者知识模型、端到端杀戮链的不完整覆盖、没有多主机和加密用户凭证的不切实际的环境以及依赖于LLM作为法官的不可靠判断。为了解决这些差距，我们提出了AdvCUA，这是MITRE ATT & CK Enterprise Matrix中第一个与现实世界TTP相一致的基准，该基准由140个任务组成，其中包括40个直接恶意任务、74个基于TTP的恶意任务和26个端到端杀死链，通过硬编码评估系统地评估多主机环境沙盒中现实企业操作系统安全威胁下的CUA。我们基于8个基础LLM评估了现有的五种主流CUA，包括ReAct、AutoGPT、Gemini CLI、Cursor CLI和Cursor IDE。结果表明，当前的前沿CUA无法充分覆盖以操作系统安全为中心的威胁。CUA的这些功能减少了对自定义恶意软件和深层领域专业知识的依赖，甚至使经验不足的攻击者也能够发动复杂的企业入侵，这引发了社会对CUA责任和安全性的担忧。



## **35. Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?**

LLM的内层是否揭示了越狱检测的模式？ cs.CL

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06594v1) [paper-pdf](http://arxiv.org/pdf/2510.06594v1)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense.

摘要: 随着对话式LLM的日益普及和可访问性，越狱大型语言模型（LLM）已成为一个紧迫的问题。敌对用户经常通过精心设计的提示来利用这些模型来获取受限或敏感的输出，这种策略被广泛称为越狱。虽然已经提出了许多防御机制，但攻击者不断开发新颖的提示技术，并且没有任何现有模型可以被认为是完全抵抗的。在这项研究中，我们通过检查LLM的内部表示来调查越狱现象，重点关注隐藏层如何对越狱与良性提示做出反应。具体来说，我们分析了开源LLM GPT-J和状态空间模型Mamba 2，提供了突出不同分层行为的初步发现。我们的结果为进一步研究利用内部模型动态来进行稳健的越狱检测和防御指明了有希望的方向。



## **36. MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks**

MM-PoisonRAG：通过本地和全球中毒攻击扰乱多模式RAG cs.LG

Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2502.17832v3) [paper-pdf](http://arxiv.org/pdf/2502.17832v3)

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-Wei Chang, Daniel Kang, Heng Ji

**Abstract**: Multimodal large language models with Retrieval Augmented Generation (RAG) have significantly advanced tasks such as multimodal question answering by grounding responses in external text and images. This grounding improves factuality, reduces hallucination, and extends reasoning beyond parametric knowledge. However, this reliance on external knowledge poses a critical yet underexplored safety risk: knowledge poisoning attacks, where adversaries deliberately inject adversarial multimodal content into external knowledge bases to steer model toward generating incorrect or even harmful responses. To expose such vulnerabilities, we propose MM-PoisonRAG, the first framework to systematically design knowledge poisoning in multimodal RAG. We introduce two complementary attack strategies: Localized Poisoning Attack (LPA), which implants targeted multimodal misinformation to manipulate specific queries, and Globalized Poisoning Attack (GPA), which inserts a single adversarial knowledge to broadly disrupt reasoning and induce nonsensical responses across all queries. Comprehensive experiments across tasks, models, and access settings show that LPA achieves targeted manipulation with attack success rates of up to 56%, while GPA completely disrupts model generation to 0% accuracy with just a single adversarial knowledge injection. Our results reveal the fragility of multimodal RAG and highlight the urgent need for defenses against knowledge poisoning.

摘要: 具有检索增强生成（RAG）的多模态大型语言模型具有显着的高级任务，例如通过外部文本和图像中的基础响应进行多模态问题回答。这种基础提高了真实性，减少了幻觉，并将推理扩展到参数知识之外。然而，这种对外部知识的依赖带来了一个关键但尚未得到充分研究的安全风险：知识中毒攻击，其中对手故意将对抗性多模态内容注入外部知识库，以引导模型生成错误甚至有害的响应。为了暴露这些漏洞，我们提出了MM-PoisonRAG，第一个框架，系统地设计知识中毒的多模式RAG。我们介绍了两种互补的攻击策略：局部中毒攻击（LPA），它植入有针对性的多模态错误信息来操纵特定的查询，和全局中毒攻击（GPA），它插入一个单一的对抗性知识来广泛地破坏推理，并在所有查询中诱导无意义的响应。跨任务、模型和访问设置的综合实验表明，LPA实现了有针对性的操作，攻击成功率高达56%，而GPA仅用一次对抗性知识注入就完全破坏了模型生成，准确率为0%。我们的研究结果揭示了多模态RAG的脆弱性，并强调了迫切需要防御知识中毒。



## **37. From Description to Detection: LLM based Extendable O-RAN Compliant Blind DoS Detection in 5G and Beyond**

从描述到检测：5G及更高版本中基于LLM的可扩展O-RAN兼容盲DPS检测 cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06530v1) [paper-pdf](http://arxiv.org/pdf/2510.06530v1)

**Authors**: Thusitha Dayaratne, Ngoc Duy Pham, Viet Vo, Shangqi Lai, Sharif Abuadbba, Hajime Suzuki, Xingliang Yuan, Carsten Rudolph

**Abstract**: The quality and experience of mobile communication have significantly improved with the introduction of 5G, and these improvements are expected to continue beyond the 5G era. However, vulnerabilities in control-plane protocols, such as Radio Resource Control (RRC) and Non-Access Stratum (NAS), pose significant security threats, such as Blind Denial of Service (DoS) attacks. Despite the availability of existing anomaly detection methods that leverage rule-based systems or traditional machine learning methods, these methods have several limitations, including the need for extensive training data, predefined rules, and limited explainability. Addressing these challenges, we propose a novel anomaly detection framework that leverages the capabilities of Large Language Models (LLMs) in zero-shot mode with unordered data and short natural language attack descriptions within the Open Radio Access Network (O-RAN) architecture. We analyse robustness to prompt variation, demonstrate the practicality of automating the attack descriptions and show that detection quality relies on the semantic completeness of the description rather than its phrasing or length. We utilise an RRC/NAS dataset to evaluate the solution and provide an extensive comparison of open-source and proprietary LLM implementations to demonstrate superior performance in attack detection. We further validate the practicality of our framework within O-RAN's real-time constraints, illustrating its potential for detecting other Layer-3 attacks.

摘要: 随着5G的推出，移动通信的质量和体验得到了显着的改善，预计这些改善将持续到5G时代之后。然而，无线电资源控制（RNC）和非访问层（NAS）等控制平面协议中的漏洞构成了严重的安全威胁，例如盲目拒绝服务（NOS）攻击。尽管现有的异常检测方法可以利用基于规则的系统或传统的机器学习方法，但这些方法存在一些局限性，包括需要大量的训练数据、预定义的规则和有限的解释性。为了应对这些挑战，我们提出了一种新颖的异常检测框架，该框架利用零触发模式下的大型语言模型（LLM）的能力，具有开放无线电接入网络（O-RAN）架构中的无序数据和简短的自然语言攻击描述。我们分析了对提示变异的鲁棒性，证明了攻击描述自动化的实用性，并表明检测质量取决于描述的语义完整性，而不是其措辞或长度。我们利用RR/NAS数据集来评估解决方案，并提供开源和专有LLM实施的广泛比较，以展示攻击检测方面的卓越性能。我们进一步验证了我们的框架在O-RAN的实时约束下的实用性，说明了其检测其他第3层攻击的潜力。



## **38. Text-to-Image Models Leave Identifiable Signatures: Implications for Leaderboard Security**

文本到图像模型留下可识别签名：对排行榜安全性的影响 cs.LG

Accepted at Lock-LLM Workshop, NeurIPS 2025

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.06525v1) [paper-pdf](http://arxiv.org/pdf/2510.06525v1)

**Authors**: Ali Naseh, Anshuman Suri, Yuefeng Peng, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Generative AI leaderboards are central to evaluating model capabilities, but remain vulnerable to manipulation. Among key adversarial objectives is rank manipulation, where an attacker must first deanonymize the models behind displayed outputs -- a threat previously demonstrated and explored for large language models (LLMs). We show that this problem can be even more severe for text-to-image leaderboards, where deanonymization is markedly easier. Using over 150,000 generated images from 280 prompts and 19 diverse models spanning multiple organizations, architectures, and sizes, we demonstrate that simple real-time classification in CLIP embedding space identifies the generating model with high accuracy, even without prompt control or historical data. We further introduce a prompt-level separability metric and identify prompts that enable near-perfect deanonymization. Our results indicate that rank manipulation in text-to-image leaderboards is easier than previously recognized, underscoring the need for stronger defenses.

摘要: 生成性人工智能排行榜是评估模型能力的核心，但仍然容易受到操纵。关键的对抗目标之一是排名操纵，攻击者必须首先对显示输出背后的模型进行去匿名化--这是之前针对大型语言模型（LLM）演示和探索的威胁。我们表明，对于文本到图像排行榜来说，这个问题可能会更加严重，因为其中的去匿名化明显更容易。使用从280个提示生成的超过150，000个图像和跨越多种组织、架构和规模的19个不同模型，我们证明了CLIP嵌入空间中的简单实时分类可以高准确性地识别生成模型，即使没有提示控制或历史数据。我们进一步引入预算级别的可分离性指标并识别能够实现近乎完美的去匿名化的提示。我们的结果表明，文本到图像排行榜中的排名操纵比之前认识到的更容易，这凸显了对更强防御的必要性。



## **39. An Embarrassingly Simple Defense Against LLM Abliteration Attacks**

针对LLM删节攻击的令人尴尬的简单防御 cs.CL

preprint - under review

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2505.19056v2) [paper-pdf](http://arxiv.org/pdf/2505.19056v2)

**Authors**: Harethah Abu Shairah, Hasan Abed Al Kader Hammoud, Bernard Ghanem, George Turkiyyah

**Abstract**: Large language models (LLMs) are typically aligned to refuse harmful instructions through safety fine-tuning. A recent attack, termed abliteration, identifies and suppresses the single latent direction most responsible for refusal behavior, thereby enabling models to generate harmful content. We propose a defense that fundamentally alters how models express refusal. We construct an extended-refusal dataset in which responses to harmful prompts provide detailed justifications before refusing, distributing the refusal signal across multiple token positions. Fine-tuning Llama-2-7B-Chat and Qwen2.5-Instruct (1.5B and 3B parameters) on this dataset yields models that maintain high refusal rates under abliteration: refusal rates drop by at most 10%, compared to 70-80% drops in baseline models. Comprehensive evaluations of safety and utility demonstrate that extended-refusal fine-tuning effectively neutralizes abliteration attacks while preserving general model performance and enhancing robustness across multiple alignment scenarios.

摘要: 大型语言模型（LLM）通常会通过安全微调来拒绝有害指令。最近的一种名为“取消”的攻击可以识别和抑制对拒绝行为负有最大责任的单一潜在方向，从而使模型能够生成有害内容。我们提出了一种从根本上改变模型表达拒绝的方式的辩护。我们构建了一个扩展拒绝数据集，其中对有害提示的响应在拒绝之前提供详细的理由，将拒绝信号分布在多个代币位置上。对该数据集进行微调Llama-2- 7 B-Chat和Qwen 2.5-Direct（1.5B和3B参数）产生的模型在更新下保持高拒绝率：拒绝率最多下降10%，而基线模型中下降了70-80%。对安全性和实用性的全面评估表明，扩展拒绝微调可以有效地中和稳定攻击，同时保留一般模型性能并增强多个对齐场景的鲁棒性。



## **40. SAFER: Advancing Safety Alignment via Efficient Ex-Ante Reasoning**

更安全：通过高效的前前推理推进安全一致 cs.CL

22 pages, 5 figures

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2504.02725v2) [paper-pdf](http://arxiv.org/pdf/2504.02725v2)

**Authors**: Kehua Feng, Keyan Ding, Yuhao Wang, Menghan Li, Fanjunduo Wei, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose SAFER, a framework for Safety Alignment via eFficient Ex-Ante Reasoning. Our approach instantiates structured Ex-Ante reasoning through initial assessment, rule verification, and path calibration, and embeds predefined safety rules to provide transparent and verifiable safety judgments. Specifically, our approach consists of two training stages: (1) supervised fine-tuning with synthetic traces to teach the multi-stage Ex-Ante reasoning, and (2) step-level reasoning preference optimization to jointly enhance safety, utility, and efficiency. Experiments on multiple open-source LLMs demonstrate that SAFER significantly enhances safety performance while maintaining helpfulness and response efficiency.

摘要: 大型语言模型（LLM）的最新进展加速了人工通用智能的发展，但它们生成有害内容的潜力带来了严峻的安全挑战。现有的对齐方法通常难以覆盖不同的安全场景，并且仍然容易受到对抗攻击。在这项工作中，我们提出了SAGER，这是一个通过eFficient Ex-Ante Reasoning进行安全调整的框架。我们的方法通过初始评估、规则验证和路径校准来实例化结构化的Ex-Ante推理，并嵌入预定义的安全规则以提供透明且可验证的安全判断。具体来说，我们的方法由两个训练阶段组成：（1）使用合成轨迹进行监督微调，以教授多阶段Ex-Ante推理，以及（2）分步推理偏好优化，以共同增强安全性、实用性和效率。对多个开源LLM的实验表明，SAGER显着增强了安全性能，同时保持了帮助性和响应效率。



## **41. Geometry-Guided Adversarial Prompt Detection via Curvature and Local Intrinsic Dimension**

通过弯曲和局部固有维度的几何引导对抗提示检测 cs.CL

40 Pages, 6 figues

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2503.03502v2) [paper-pdf](http://arxiv.org/pdf/2503.03502v2)

**Authors**: Canaan Yung, Hanxun Huang, Christopher Leckie, Sarah Erfani

**Abstract**: Adversarial prompts are capable of jailbreaking frontier large language models (LLMs) and inducing undesirable behaviours, posing a significant obstacle to their safe deployment. Current mitigation strategies primarily rely on activating built-in defence mechanisms or fine-tuning LLMs, both of which are computationally expensive and can sacrifice model utility. In contrast, detection-based approaches are more efficient and practical for deployment in real-world applications. However, the fundamental distinctions between adversarial and benign prompts remain poorly understood. In this work, we introduce CurvaLID, a novel defence framework that efficiently detects adversarial prompts by leveraging their geometric properties. It is agnostic to the type of LLM, offering a unified detection framework across diverse adversarial prompts and LLM architectures. CurvaLID builds on the geometric analysis of text prompts to uncover their underlying differences. We theoretically extend the concept of curvature via the Whewell equation into an $n$-dimensional word embedding space, enabling us to quantify local geometric properties, including semantic shifts and curvature in the underlying manifolds. To further enhance our solution, we leverage Local Intrinsic Dimensionality (LID) to capture complementary geometric features of text prompts within adversarial subspaces. Our findings show that adversarial prompts exhibit distinct geometric signatures from benign prompts, enabling CurvaLID to achieve near-perfect classification and outperform state-of-the-art detectors in adversarial prompt detection. CurvaLID provides a reliable and efficient safeguard against malicious queries as a model-agnostic method that generalises across multiple LLMs and attack families.

摘要: 对抗性提示能够越狱前沿大型语言模型（LLM）并引发不良行为，对其安全部署构成重大障碍。当前的缓解策略主要依赖于激活内置防御机制或微调LLM，这两者计算成本很高，并且可能会牺牲模型效用。相比之下，基于检测的方法对于在现实世界应用程序中部署更有效和实用。然而，对抗性提示和良性提示之间的根本区别仍然知之甚少。在这项工作中，我们引入了CurvaLID，这是一种新型防御框架，可以通过利用其几何属性来有效检测对抗提示。它与LLM类型无关，提供跨不同对抗提示和LLM架构的统一检测框架。CurvaLID基于文本提示的几何分析，以揭示其潜在差异。从理论上讲，我们通过Whewell方程将弯曲的概念扩展到$n维单词嵌入空间，使我们能够量化局部几何属性，包括底层流中的语义移动和弯曲。为了进一步增强我们的解决方案，我们利用局部本质模糊性（LID）来捕获对抗子空间中文本提示的补充几何特征。我们的研究结果表明，对抗性提示表现出与良性提示不同的几何特征，使CurvaLID能够实现近乎完美的分类，并在对抗性提示检测方面优于最先进的检测器。CurvaLID作为一种模型不可知的方法，可在多个LLM和攻击系列中推广，提供可靠且高效的防范恶意查询的保护措施。



## **42. Towards Reliable and Practical LLM Security Evaluations via Bayesian Modelling**

通过Bayesian建模实现可靠且实用的LLM安全评估 cs.CR

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05709v1) [paper-pdf](http://arxiv.org/pdf/2510.05709v1)

**Authors**: Mary Llewellyn, Annie Gray, Josh Collyer, Michael Harries

**Abstract**: Before adopting a new large language model (LLM) architecture, it is critical to understand vulnerabilities accurately. Existing evaluations can be difficult to trust, often drawing conclusions from LLMs that are not meaningfully comparable, relying on heuristic inputs or employing metrics that fail to capture the inherent uncertainty. In this paper, we propose a principled and practical end-to-end framework for evaluating LLM vulnerabilities to prompt injection attacks. First, we propose practical approaches to experimental design, tackling unfair LLM comparisons by considering two practitioner scenarios: when training an LLM and when deploying a pre-trained LLM. Second, we address the analysis of experiments and propose a Bayesian hierarchical model with embedding-space clustering. This model is designed to improve uncertainty quantification in the common scenario that LLM outputs are not deterministic, test prompts are designed imperfectly, and practitioners only have a limited amount of compute to evaluate vulnerabilities. We show the improved inferential capabilities of the model in several prompt injection attack settings. Finally, we demonstrate the pipeline to evaluate the security of Transformer versus Mamba architectures. Our findings show that consideration of output variability can suggest less definitive findings. However, for some attacks, we find notably increased Transformer and Mamba-variant vulnerabilities across LLMs with the same training data or mathematical ability.

摘要: 在采用新的大型语言模型（LLM）架构之前，准确了解漏洞至关重要。现有的评估可能很难信任，通常从没有意义可比性的LLM中得出结论，依赖于启发式输入或采用未能捕捉固有不确定性的指标。在本文中，我们提出了一个原则性且实用的端到端框架，用于评估LLM漏洞以引发注入攻击。首先，我们提出了实验设计的实用方法，通过考虑两种从业者场景来解决不公平的LLM比较：当训练LLM时和当部署预训练的LLM时。其次，我们进行了实验分析，并提出了一个具有嵌入空间集群的Bayesian分层模型。该模型旨在在LLM输出不确定性、测试提示设计不完美以及从业者只有有限的计算量来评估漏洞等常见场景中改进不确定性量化。我们展示了该模型在几种即时注入攻击设置中改进的推理能力。最后，我们演示了评估Transformer与Mamba架构安全性的管道。我们的研究结果表明，考虑输出变异性可能会得出不太确定的结果。然而，对于某些攻击，我们发现具有相同训练数据或数学能力的LLM中Transformer和Mamba变体漏洞显着增加。



## **43. Membership Inference Attacks on Tokenizers of Large Language Models**

对大型语言模型令牌器的成员推断攻击 cs.CR

Code is available at: https://github.com/mengtong0110/Tokenizer-MIA

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05699v1) [paper-pdf](http://arxiv.org/pdf/2510.05699v1)

**Authors**: Meng Tong, Yuntao Du, Kejiang Chen, Weiming Zhang, Ninghui Li

**Abstract**: Membership inference attacks (MIAs) are widely used to assess the privacy risks associated with machine learning models. However, when these attacks are applied to pre-trained large language models (LLMs), they encounter significant challenges, including mislabeled samples, distribution shifts, and discrepancies in model size between experimental and real-world settings. To address these limitations, we introduce tokenizers as a new attack vector for membership inference. Specifically, a tokenizer converts raw text into tokens for LLMs. Unlike full models, tokenizers can be efficiently trained from scratch, thereby avoiding the aforementioned challenges. In addition, the tokenizer's training data is typically representative of the data used to pre-train LLMs. Despite these advantages, the potential of tokenizers as an attack vector remains unexplored. To this end, we present the first study on membership leakage through tokenizers and explore five attack methods to infer dataset membership. Extensive experiments on millions of Internet samples reveal the vulnerabilities in the tokenizers of state-of-the-art LLMs. To mitigate this emerging risk, we further propose an adaptive defense. Our findings highlight tokenizers as an overlooked yet critical privacy threat, underscoring the urgent need for privacy-preserving mechanisms specifically designed for them.

摘要: 成员资格推理攻击（MIA）广泛用于评估与机器学习模型相关的隐私风险。然而，当这些攻击应用于预训练的大型语言模型（LLM）时，它们会遇到重大挑战，包括样本标签错误、分布变化以及实验环境和现实环境之间模型大小的差异。为了解决这些限制，我们引入了标记器作为成员资格推断的新攻击载体。具体来说，标记器将原始文本转换为LLM的标记。与完整模型不同，标记器可以从头开始有效训练，从而避免上述挑战。此外，标记化器的训练数据通常代表用于预训练LLM的数据。尽管有这些优势，但标记器作为攻击载体的潜力仍未被开发。为此，我们提出了第一项关于通过标记器的成员资格泄露的研究，并探索了五种推断数据集成员资格的攻击方法。对数百万个互联网样本的广泛实验揭示了最先进的LLM标记器中的漏洞。为了减轻这种新出现的风险，我们进一步提出了适应性防御。我们的研究结果强调，代币使用者是一种被忽视但又严重的隐私威胁，强调迫切需要专门为它们设计的隐私保护机制。



## **44. Bypassing Prompt Guards in Production with Controlled-Release Prompting**

通过控制释放注射来消除生产中的及时防护 cs.LG

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.01529v2) [paper-pdf](http://arxiv.org/pdf/2510.01529v2)

**Authors**: Jaiden Fairoze, Sanjam Garg, Keewoo Lee, Mingyuan Wang

**Abstract**: As large language models (LLMs) advance, ensuring AI safety and alignment is paramount. One popular approach is prompt guards, lightweight mechanisms designed to filter malicious queries while being easy to implement and update. In this work, we introduce a new attack that circumvents such prompt guards, highlighting their limitations. Our method consistently jailbreaks production models while maintaining response quality, even under the highly protected chat interfaces of Google Gemini (2.5 Flash/Pro), DeepSeek Chat (DeepThink), Grok (3), and Mistral Le Chat (Magistral). The attack exploits a resource asymmetry between the prompt guard and the main LLM, encoding a jailbreak prompt that lightweight guards cannot decode but the main model can. This reveals an attack surface inherent to lightweight prompt guards in modern LLM architectures and underscores the need to shift defenses from blocking malicious inputs to preventing malicious outputs. We additionally identify other critical alignment issues, such as copyrighted data extraction, training data extraction, and malicious response leakage during thinking.

摘要: 随着大型语言模型（LLM）的发展，确保人工智能的安全性和一致性至关重要。一种流行的方法是提示保护，这是一种轻量级机制，旨在过滤恶意查询，同时易于实现和更新。在这项工作中，我们引入了一种新的攻击，可以绕过此类提示警卫，强调了它们的局限性。即使在Google Gemini（2.5 Flash/Pro）、DeepSeek Chat（DeepThink）、Grok（3）和Mistral Le Chat（Magistral）等受到高度保护的聊天界面下，我们的方法也能始终如一地越狱生产模型，同时保持响应质量。该攻击利用提示守卫和主LLM之间的资源不对称，编码轻量级守卫无法解码但主模型可以解码的越狱提示。这揭示了现代LLM架构中轻量级提示保护固有的攻击表面，并强调了将防御从阻止恶意输入转向阻止恶意输出的必要性。我们还识别了其他关键的对齐问题，例如受版权保护的数据提取、训练数据提取和思考期间的恶意响应泄露。



## **45. AutoPentester: An LLM Agent-based Framework for Automated Pentesting**

AutoPentester：一个基于LLM代理的自动Penttesting框架 cs.CR

IEEE TrustCom 2025 10 pages

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05605v1) [paper-pdf](http://arxiv.org/pdf/2510.05605v1)

**Authors**: Yasod Ginige, Akila Niroshan, Sajal Jain, Suranga Seneviratne

**Abstract**: Penetration testing and vulnerability assessment are essential industry practices for safeguarding computer systems. As cyber threats grow in scale and complexity, the demand for pentesting has surged, surpassing the capacity of human professionals to meet it effectively. With advances in AI, particularly Large Language Models (LLMs), there have been attempts to automate the pentesting process. However, existing tools such as PentestGPT are still semi-manual, requiring significant professional human interaction to conduct pentests. To this end, we propose a novel LLM agent-based framework, AutoPentester, which automates the pentesting process. Given a target IP, AutoPentester automatically conducts pentesting steps using common security tools in an iterative process. It can dynamically generate attack strategies based on the tool outputs from the previous iteration, mimicking the human pentester approach. We evaluate AutoPentester using Hack The Box and custom-made VMs, comparing the results with the state-of-the-art PentestGPT. Results show that AutoPentester achieves a 27.0% better subtask completion rate and 39.5% more vulnerability coverage with fewer steps. Most importantly, it requires significantly fewer human interactions and interventions compared to PentestGPT. Furthermore, we recruit a group of security industry professional volunteers for a user survey and perform a qualitative analysis to evaluate AutoPentester against industry practices and compare it with PentestGPT. On average, AutoPentester received a score of 3.93 out of 5 based on user reviews, which was 19.8% higher than PentestGPT.

摘要: 渗透测试和漏洞评估是保护计算机系统的重要行业实践。随着网络威胁规模和复杂性的增长，对冥想的需求激增，超出了人类专业人士有效应对的能力。随着人工智能，特别是大型语言模型（LLM）的进步，人们已经尝试自动化铅笔测试过程。然而，PentestGPT等现有工具仍然是半手动的，需要大量的专业人员互动才能进行PentestGPT。为此，我们提出了一种新型的LLM基于代理的框架AutoPentester，它可以自动化Penttester。给定目标IP，AutoPentester在迭代过程中使用常用安全工具自动执行Penttesting步骤。它可以根据上一次迭代的工具输出动态生成攻击策略，模仿人类penttester方法。我们使用Hack The Box和定制虚拟机来评估AutoPentester，并将结果与最先进的PentestGPT进行比较。结果表明，AutoPentester子任务完成率提高了27.0%，漏洞覆盖率提高了39.5%，步骤更少。最重要的是，与PentestGPT相比，它需要更少的人类互动和干预。此外，我们招募了一批安全行业的专业志愿者进行用户调查，并进行定性分析，以评估AutoPentester与行业惯例，并将其与PentestGPT进行比较。根据用户评论，AutoPentester平均获得3.93分（满分5分），比PentestGPT高出19.8%。



## **46. A Middle Path for On-Premises LLM Deployment: Preserving Privacy Without Sacrificing Model Confidentiality**

本地LLM部署的中间路径：在不牺牲模型机密性的情况下保护隐私 cs.LG

8 pages for main content of the paper

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2410.11182v3) [paper-pdf](http://arxiv.org/pdf/2410.11182v3)

**Authors**: Hanbo Huang, Yihan Li, Bowen Jiang, Bo Jiang, Lin Liu, Ruoyu Sun, Zhuotao Liu, Shiyu Liang

**Abstract**: Privacy-sensitive users require deploying large language models (LLMs) within their own infrastructure (on-premises) to safeguard private data and enable customization. However, vulnerabilities in local environments can lead to unauthorized access and potential model theft. To address this, prior research on small models has explored securing only the output layer within hardware-secured devices to balance model confidentiality and customization. Yet this approach fails to protect LLMs effectively. In this paper, we discover that (1) query-based distillation attacks targeting the secured top layer can produce a functionally equivalent replica of the victim model; (2) securing the same number of layers, bottom layers before a transition layer provide stronger protection against distillation attacks than top layers, with comparable effects on customization performance; and (3) the number of secured layers creates a trade-off between protection and customization flexibility. Based on these insights, we propose SOLID, a novel deployment framework that secures a few bottom layers in a secure environment and introduces an efficient metric to optimize the trade-off by determining the ideal number of hidden layers. Extensive experiments on five models (1.3B to 70B parameters) demonstrate that SOLID outperforms baselines, achieving a better balance between protection and downstream customization.

摘要: 对隐私敏感的用户需要在自己的基础设施（本地）内部署大型语言模型（LLM），以保护私人数据并实现定制。然而，本地环境中的漏洞可能会导致未经授权的访问和潜在的模型盗窃。为了解决这个问题，之前对小型模型的研究探索了仅保护硬件保护设备内的输出层，以平衡模型机密性和定制性。然而这种方法未能有效保护LLC。在本文中，我们发现（1）针对受保护顶层的基于查询的蒸馏攻击可以产生受害者模型的功能等效副本;（2）在过渡层之前保护相同数量的层，底层提供了比顶层更强的针对蒸馏攻击的保护，对定制性能的影响相当;以及（3）安全层的数量在保护和定制灵活性之间产生了权衡。基于这些见解，我们提出了实体部署框架，这是一种新颖的部署框架，可以在安全环境中保护一些底层，并引入一种有效的指标来通过确定隐藏层的理想数量来优化权衡。对五个模型（1.3B至70 B参数）的广泛实验表明，SOUID优于基线，在保护和下游定制之间实现了更好的平衡。



## **47. Adversarial Reinforcement Learning for Large Language Model Agent Safety**

用于大语言模型代理安全的对抗强化学习 cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05442v1) [paper-pdf](http://arxiv.org/pdf/2510.05442v1)

**Authors**: Zizhao Wang, Dingcheng Li, Vaishakh Keshava, Phillip Wallis, Ananth Balashankar, Peter Stone, Lukas Rutishauser

**Abstract**: Large Language Model (LLM) agents can leverage tools such as Google Search to complete complex tasks. However, this tool usage introduces the risk of indirect prompt injections, where malicious instructions hidden in tool outputs can manipulate the agent, posing security risks like data leakage. Current defense strategies typically rely on fine-tuning LLM agents on datasets of known attacks. However, the generation of these datasets relies on manually crafted attack patterns, which limits their diversity and leaves agents vulnerable to novel prompt injections. To address this limitation, we propose Adversarial Reinforcement Learning for Agent Safety (ARLAS), a novel framework that leverages adversarial reinforcement learning (RL) by formulating the problem as a two-player zero-sum game. ARLAS co-trains two LLMs: an attacker that learns to autonomously generate diverse prompt injections and an agent that learns to defend against them while completing its assigned tasks. To ensure robustness against a wide range of attacks and to prevent cyclic learning, we employ a population-based learning framework that trains the agent to defend against all previous attacker checkpoints. Evaluated on BrowserGym and AgentDojo, agents fine-tuned with ARLAS achieve a significantly lower attack success rate than the original model while also improving their task success rate. Our analysis further confirms that the adversarial process generates a diverse and challenging set of attacks, leading to a more robust agent compared to the base model.

摘要: 大型语言模型（LLM）代理可以利用Google Search等工具来完成复杂的任务。然而，这种工具的使用引入了间接提示注入的风险，其中隐藏在工具输出中的恶意指令可以操纵代理，从而带来数据泄露等安全风险。当前的防御策略通常依赖于对已知攻击数据集进行微调LLM代理。然而，这些数据集的生成依赖于手动设计的攻击模式，这限制了它们的多样性，并使代理容易受到新型提示注入的影响。为了解决这一局限性，我们提出了针对代理安全的对抗强化学习（ARLAS），这是一个新颖的框架，通过将问题表述为两人零和游戏来利用对抗强化学习（RL）。ARLAS联合培训了两名LLM：攻击者学会自主生成各种提示注入，而代理则学会在完成分配的任务的同时防御它们。为了确保针对广泛攻击的鲁棒性并防止循环学习，我们采用了基于群体的学习框架，该框架训练代理抵御所有之前的攻击者检查点。在BrowserGym和AgentDojo上进行评估，使用ARLAS微调的代理比原始模型实现了显着降低的攻击成功率，同时也提高了任务成功率。我们的分析进一步证实，对抗过程会产生一系列多样化且具有挑战性的攻击，从而导致与基本模型相比更强大的代理。



## **48. DP-Adam-AC: Privacy-preserving Fine-Tuning of Localizable Language Models Using Adam Optimization with Adaptive Clipping**

DP-Adam-AC：使用Adam优化和自适应剪辑对可本地化语言模型进行隐私保护微调 cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05288v1) [paper-pdf](http://arxiv.org/pdf/2510.05288v1)

**Authors**: Ruoxing Yang

**Abstract**: Large language models (LLMs) such as ChatGPT have evolved into powerful and ubiquitous tools. Fine-tuning on small datasets allows LLMs to acquire specialized skills for specific tasks efficiently. Although LLMs provide great utility in both general and task-specific use cases, they are limited by two security-related concerns. First, traditional LLM hardware requirements make them infeasible to run locally on consumer-grade devices. A remote network connection with the LLM provider's server is usually required, making the system vulnerable to network attacks. Second, fine-tuning an LLM for a sensitive task may involve sensitive data. Non-private fine-tuning algorithms produce models vulnerable to training data reproduction attacks. Our work addresses these security concerns by enhancing differentially private optimization algorithms and applying them to fine-tune localizable language models. We introduce adaptable gradient clipping along with other engineering enhancements to the standard DP-Adam optimizer to create DP-Adam-AC. We use our optimizer to fine-tune examples of two localizable LLM designs, small language model (Qwen2.5-0.5B) and 1.58 bit quantization (Bitnet-b1.58-2B). We demonstrate promising improvements in loss through experimentation with two synthetic datasets.

摘要: ChatGPT等大型语言模型（LLM）已发展成为强大且无处不在的工具。对小型数据集的微调使LLM能够有效地获得特定任务的专业技能。尽管LLM在一般和特定任务用例中都提供了很大的实用性，但它们受到两个安全相关问题的限制。首先，传统的LLM硬件要求使它们无法在消费级设备上本地运行。通常需要与LLM提供商的服务器建立远程网络连接，这使得系统容易受到网络攻击。其次，针对敏感任务微调LLM可能涉及敏感数据。非私有微调算法产生的模型容易受到训练数据复制攻击。我们的工作通过增强差异私密优化算法并将其应用于微调可本地化语言模型来解决这些安全问题。我们对标准DP-Adam优化器引入了可适应的梯度剪裁以及其他工程增强，以创建DP-Adam-AC。我们使用优化器来微调两种可本地化LLM设计的示例，即小型语言模型（Qwen 2.5 -0.5B）和1.58位量化（Bitnet-b1.58-2B）。我们通过对两个合成数据集的实验证明了损失方面的有希望的改善。



## **49. Proactive defense against LLM Jailbreak**

针对LLM越狱的积极防御 cs.CR

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05052v1) [paper-pdf](http://arxiv.org/pdf/2510.05052v1)

**Authors**: Weiliang Zhao, Jinjun Peng, Daniel Ben-Levi, Zhou Yu, Junfeng Yang

**Abstract**: The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks.

摘要: 强大的大型语言模型（LLM）的激增需要强大的安全对齐，但这些模型仍然容易受到不断发展的对抗攻击，包括迭代搜索成功查询的多回合越狱。当前的防御措施主要是反应性和静态的，通常无法抵御这些基于搜索的攻击。在本文中，我们介绍了ProAct，这是一种新颖的主动防御框架，旨在扰乱和误导自主越狱过程。我们的核心想法是故意向对手提供“虚假回应”，这些回应似乎是成功越狱攻击的结果，但不包含实际的有害内容。这些误导性响应为攻击者的内部优化循环提供了错误信号，导致对抗性搜索提前终止并有效地越狱。通过在最先进的LLM、越狱框架和安全基准上进行广泛的实验，我们的方法持续且显着地将攻击成功率降低高达92%。与其他防御框架结合使用时，它进一步将最新攻击策略的成功率降低至0%。ProAct代表了一种垂直防御策略，可以作为额外的护栏，以增强LLM的安全性，抵御最有效的越狱攻击。



## **50. Rethinking Exact Unlearning under Exposure: Extracting Forgotten Data under Exact Unlearning in Large Language Model**

重新思考暴露下的精确取消学习：大型语言模型中精确取消学习下的被遗忘数据 cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2505.24379v2) [paper-pdf](http://arxiv.org/pdf/2505.24379v2)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.

摘要: 大型语言模型通常在从网络收集的数据集上进行训练，这些数据集可能无意中包含有害或敏感的个人信息。为了解决日益增长的隐私问题，人们提出了取消学习方法来消除训练模型中特定数据的影响。其中，精确的取消学习（在没有目标数据的情况下从头开始重新训练模型）被广泛认为是减轻部署中隐私风险的黄金标准。在本文中，我们在实际部署环境中重新审视了这一假设，其中暴露了取消学习前和取消学习后的日志API，例如在开放重量场景中。针对此设置，我们引入了一种新颖的数据提取攻击，该攻击利用来自取消学习前模型的信号来指导取消学习后模型，从而发现反映已删除数据分布的模式。将模型指导与令牌过滤策略相结合，我们的攻击显着提高了MUSE、TOFU和WMDP等常见基准测试中的提取成功率，在某些情况下性能翻倍。此外，我们还展示了我们对模拟医疗诊断数据集的攻击的有效性，以强调与精确忘记相关的现实世界隐私风险。我们的研究结果表明，取消学习可能会以一种矛盾的方式增加现实世界部署期间隐私泄露的风险，鉴于此，我们主张评估取消学习方法，以考虑更广泛的威胁模型，这些模型不仅考虑取消学习后的模型，还考虑到对之前检查点的对抗访问。代码可在https://github.com/Nicholas0228/unlearned_data_extraction_llm上公开获取。




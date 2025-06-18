# Latest Large Language Model Attack Papers
**update at 2025-06-18 11:05:27**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AIRTBench: Measuring Autonomous AI Red Teaming Capabilities in Language Models**

AIRTBench：衡量语言模型中的自主AI Red团队能力 cs.CR

43 pages, 13 figures, 16 tables

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14682v1) [paper-pdf](http://arxiv.org/pdf/2506.14682v1)

**Authors**: Ads Dawson, Rob Mulla, Nick Landers, Shane Caldwell

**Abstract**: We introduce AIRTBench, an AI red teaming benchmark for evaluating language models' ability to autonomously discover and exploit Artificial Intelligence and Machine Learning (AI/ML) security vulnerabilities. The benchmark consists of 70 realistic black-box capture-the-flag (CTF) challenges from the Crucible challenge environment on the Dreadnode platform, requiring models to write python code to interact with and compromise AI systems. Claude-3.7-Sonnet emerged as the clear leader, solving 43 challenges (61% of the total suite, 46.9% overall success rate), with Gemini-2.5-Pro following at 39 challenges (56%, 34.3% overall), GPT-4.5-Preview at 34 challenges (49%, 36.9% overall), and DeepSeek R1 at 29 challenges (41%, 26.9% overall). Our evaluations show frontier models excel at prompt injection attacks (averaging 49% success rates) but struggle with system exploitation and model inversion challenges (below 26%, even for the best performers). Frontier models are far outpacing open-source alternatives, with the best truly open-source model (Llama-4-17B) solving 7 challenges (10%, 1.0% overall), though demonstrating specialized capabilities on certain hard challenges. Compared to human security researchers, large language models (LLMs) solve challenges with remarkable efficiency completing in minutes what typically takes humans hours or days-with efficiency advantages of over 5,000x on hard challenges. Our contribution fills a critical gap in the evaluation landscape, providing the first comprehensive benchmark specifically designed to measure and track progress in autonomous AI red teaming capabilities.

摘要: 我们引入AIRTBench，这是一个人工智能红色团队基准，用于评估语言模型自主发现和利用人工智能和机器学习（AI/ML）安全漏洞的能力。该基准测试由来自Dreadnote平台上Crucible挑战环境的70个真实黑匣子捕获旗帜（CTF）挑战组成，要求模型编写Python代码来与人工智能系统交互和危害。Claude-3.7-十四行诗成为明确的领导者，解决了43项挑战（占总套件的61%，总体成功率46.9%），Gemini-2.5-Pro紧随其后，挑战39项（56%，总体34.3%），GPT-4.5-预览34次挑战（49%，总体36.9%），DeepSeek R1 29次挑战（41%，总体26.9%）。我们的评估显示，前沿模型在即时注入攻击方面表现出色（平均成功率为49%），但在系统利用和模型倒置挑战方面遇到了困难（即使是表现最好的，也低于26%）。Frontier模型远远超过开源替代品，最好的真正开源模型（Llama-4- 17 B）解决了7项挑战（总体10%，1.0%），尽管展示了应对某些困难挑战的专业能力。与人类安全研究人员相比，大型语言模型（LLM）以非凡的效率解决挑战，可以在几分钟内完成人类通常需要数小时或数天的任务，在艰巨的挑战上效率优势超过5，000倍。我们的贡献填补了评估领域的一个关键空白，提供了第一个专门用于衡量和跟踪自主人工智能红色团队能力进展的全面基准。



## **2. IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems**

针对基于LLM的多代理系统的IP泄露攻击 cs.CR

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2505.12442v3) [paper-pdf](http://arxiv.org/pdf/2505.12442v3)

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.

摘要: 大型语言模型（LLM）的快速发展导致了通过协作执行复杂任务的多智能体系统（MAS）的出现。然而，MAS的复杂性质，包括其架构和代理交互，引发了有关知识产权（IP）保护的严重担忧。本文介绍MASLEAK，这是一种新型攻击框架，旨在从MAS应用程序中提取敏感信息。MASLEAK针对的是实用的黑匣子设置，其中对手不了解MAS架构或代理配置。对手只能通过其公共API与MAS交互，提交攻击查询$q$并观察最终代理的输出。受计算机蠕虫传播和感染脆弱网络主机的方式的启发，MASLEAK精心设计了对抗性查询$q$，以引发、传播和保留每个MAS代理的响应，这些响应揭示了全套专有组件，包括代理数量、系统布局、系统提示、任务指令和工具使用。我们构建了包含810个应用程序的第一个MAS应用程序合成数据集，并根据现实世界的MAS应用程序（包括Coze和CrewAI）评估MASLEAK。MASLEAK在提取MAS IP方面实现了高准确性，系统提示和任务指令的平均攻击成功率为87%，大多数情况下系统架构的平均攻击成功率为92%。最后，我们讨论了我们发现的影响和潜在的防御措施。



## **3. Doppelgänger Method: Breaking Role Consistency in LLM Agent via Prompt-based Transferable Adversarial Attack**

Doppelgänger方法：通过基于预算的可转移对抗攻击打破LLM代理中的角色一致性 cs.AI

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14539v1) [paper-pdf](http://arxiv.org/pdf/2506.14539v1)

**Authors**: Daewon Kang, YeongHwan Shin, Doyeon Kim, Kyu-Hwan Jung, Meong Hi Son

**Abstract**: Since the advent of large language models, prompt engineering now enables the rapid, low-effort creation of diverse autonomous agents that are already in widespread use. Yet this convenience raises urgent concerns about the safety, robustness, and behavioral consistency of the underlying prompts, along with the pressing challenge of preventing those prompts from being exposed to user's attempts. In this paper, we propose the ''Doppelg\"anger method'' to demonstrate the risk of an agent being hijacked, thereby exposing system instructions and internal information. Next, we define the ''Prompt Alignment Collapse under Adversarial Transfer (PACAT)'' level to evaluate the vulnerability to this adversarial transfer attack. We also propose a ''Caution for Adversarial Transfer (CAT)'' prompt to counter the Doppelg\"anger method. The experimental results demonstrate that the Doppelg\"anger method can compromise the agent's consistency and expose its internal information. In contrast, CAT prompts enable effective defense against this adversarial attack.

摘要: 自从大型语言模型的出现以来，即时工程现在可以快速、低努力地创建已经广泛使用的各种自治代理。然而，这种便利性引发了人们对底层提示的安全性、稳健性和行为一致性的紧迫担忧，以及防止这些提示暴露于用户尝试的紧迫挑战。在本文中，我们提出了“Doppelg”愤怒方法来演示代理被劫持从而暴露系统指令和内部信息的风险。接下来，我们定义“对抗性转移下的提示对齐崩溃（PACAT RST）”级别来评估这种对抗性转移攻击的脆弱性。我们还提出了一个“对抗性转移警告（CAT）”提示来对抗Doppelg '愤怒方法。实验结果表明，Doppelg愤怒方法会损害代理人的一致性并暴露其内部信息。相比之下，CAT提示可以有效防御这种对抗性攻击。



## **4. LingoLoop Attack: Trapping MLLMs via Linguistic Context and State Entrapment into Endless Loops**

LingoLoop攻击：通过语言背景陷阱MLLM并将国家陷入无尽循环 cs.CL

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14493v1) [paper-pdf](http://arxiv.org/pdf/2506.14493v1)

**Authors**: Jiyuan Fu, Kaixun Jiang, Lingyi Hong, Jinglun Li, Haijing Guo, Dingkang Yang, Zhaoyu Chen, Wenqiang Zhang

**Abstract**: Multimodal Large Language Models (MLLMs) have shown great promise but require substantial computational resources during inference. Attackers can exploit this by inducing excessive output, leading to resource exhaustion and service degradation. Prior energy-latency attacks aim to increase generation time by broadly shifting the output token distribution away from the EOS token, but they neglect the influence of token-level Part-of-Speech (POS) characteristics on EOS and sentence-level structural patterns on output counts, limiting their efficacy. To address this, we propose LingoLoop, an attack designed to induce MLLMs to generate excessively verbose and repetitive sequences. First, we find that the POS tag of a token strongly affects the likelihood of generating an EOS token. Based on this insight, we propose a POS-Aware Delay Mechanism to postpone EOS token generation by adjusting attention weights guided by POS information. Second, we identify that constraining output diversity to induce repetitive loops is effective for sustained generation. We introduce a Generative Path Pruning Mechanism that limits the magnitude of hidden states, encouraging the model to produce persistent loops. Extensive experiments demonstrate LingoLoop can increase generated tokens by up to 30 times and energy consumption by a comparable factor on models like Qwen2.5-VL-3B, consistently driving MLLMs towards their maximum generation limits. These findings expose significant MLLMs' vulnerabilities, posing challenges for their reliable deployment. The code will be released publicly following the paper's acceptance.

摘要: 多模式大型语言模型（MLLM）已显示出巨大的前景，但在推理过程中需要大量的计算资源。攻击者可以通过诱导过度输出来利用这一点，导致资源耗尽和服务降级。先前的能量延迟攻击旨在通过将输出令牌分布广泛地从EOS令牌转移来增加生成时间，但它们忽视了令牌级词性（POS）特征对EOS的影响以及业务级结构模式对输出计数的影响，限制了它们的功效。为了解决这个问题，我们提出了LingoLoop，这是一种旨在诱导MLLM生成过于冗长和重复的序列的攻击。首先，我们发现代币的POS标签强烈影响生成EOS代币的可能性。基于这一见解，我们提出了一种POS感知延迟机制，通过调整POS信息引导的关注权重来推迟EOS令牌的生成。其次，我们发现限制产出多样性以引发重复循环对于持续发电是有效的。我们引入了生成路径修剪机制，该机制限制隐藏状态的大小，鼓励模型产生持久循环。大量实验表明，LingoLoop可以将生成的代币增加多达30倍，将能源消耗增加到Qwen 2.5-DL-3B等模型的可比系数，从而持续推动MLLM达到最大发电极限。这些发现暴露了MLLM的重大漏洞，为其可靠部署带来了挑战。该代码将在该报接受后公开发布。



## **5. Ensemble Watermarks for Large Language Models**

大型语言模型的注册水印 cs.CL

Accepted to ACL 2025 main conference. This article extends our  earlier work arXiv:2405.08400 by introducing an ensemble of stylometric  watermarking features and alternative experimental analysis. Code and data  are available at http://github.com/CommodoreEU/ensemble-watermark

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2411.19563v2) [paper-pdf](http://arxiv.org/pdf/2411.19563v2)

**Authors**: Georg Niess, Roman Kern

**Abstract**: As large language models (LLMs) reach human-like fluency, reliably distinguishing AI-generated text from human authorship becomes increasingly difficult. While watermarks already exist for LLMs, they often lack flexibility and struggle with attacks such as paraphrasing. To address these issues, we propose a multi-feature method for generating watermarks that combines multiple distinct watermark features into an ensemble watermark. Concretely, we combine acrostica and sensorimotor norms with the established red-green watermark to achieve a 98% detection rate. After a paraphrasing attack, the performance remains high with 95% detection rate. In comparison, the red-green feature alone as a baseline achieves a detection rate of 49% after paraphrasing. The evaluation of all feature combinations reveals that the ensemble of all three consistently has the highest detection rate across several LLMs and watermark strength settings. Due to the flexibility of combining features in the ensemble, various requirements and trade-offs can be addressed. Additionally, the same detection function can be used without adaptations for all ensemble configurations. This method is particularly of interest to facilitate accountability and prevent societal harm.

摘要: 随着大型语言模型（LLM）达到类似人类的流利程度，可靠地区分人工智能生成的文本与人类作者身份变得越来越困难。虽然LLM已经存在水印，但它们通常缺乏灵活性，并且很难应对重述等攻击。为了解决这些问题，我们提出了一种用于生成水印的多特征方法，该方法将多个不同的水印特征组合成集成水印。具体来说，我们将极乐律和感觉运动规范与既定的红-绿水印相结合，实现98%的检测率。经过重述攻击后，性能仍然很高，检测率为95%。相比之下，仅以红绿色特征作为基线，重述后检测率可达49%。对所有特征组合的评估表明，这三个特征的集合在多个LLM和水印强度设置中始终具有最高的检测率。由于在集成中组合功能的灵活性，可以满足各种要求和权衡。此外，可以使用相同的检测功能，而无需对所有总体配置进行调整。这种方法对于促进问责制和防止社会危害特别有意义。



## **6. Excessive Reasoning Attack on Reasoning LLMs**

对推理LLM的过度推理攻击 cs.CR

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14374v1) [paper-pdf](http://arxiv.org/pdf/2506.14374v1)

**Authors**: Wai Man Si, Mingjie Li, Michael Backes, Yang Zhang

**Abstract**: Recent reasoning large language models (LLMs), such as OpenAI o1 and DeepSeek-R1, exhibit strong performance on complex tasks through test-time inference scaling. However, prior studies have shown that these models often incur significant computational costs due to excessive reasoning, such as frequent switching between reasoning trajectories (e.g., underthinking) or redundant reasoning on simple questions (e.g., overthinking). In this work, we expose a novel threat: adversarial inputs can be crafted to exploit excessive reasoning behaviors and substantially increase computational overhead without compromising model utility. Therefore, we propose a novel loss framework consisting of three components: (1) Priority Cross-Entropy Loss, a modification of the standard cross-entropy objective that emphasizes key tokens by leveraging the autoregressive nature of LMs; (2) Excessive Reasoning Loss, which encourages the model to initiate additional reasoning paths during inference; and (3) Delayed Termination Loss, which is designed to extend the reasoning process and defer the generation of final outputs. We optimize and evaluate our attack for the GSM8K and ORCA datasets on DeepSeek-R1-Distill-LLaMA and DeepSeek-R1-Distill-Qwen. Empirical results demonstrate a 3x to 9x increase in reasoning length with comparable utility performance. Furthermore, our crafted adversarial inputs exhibit transferability, inducing computational overhead in o3-mini, o1-mini, DeepSeek-R1, and QWQ models.

摘要: 最近的推理大型语言模型（LLM），例如OpenAI o 1和DeepSeek-R1，通过测试时推理扩展，在复杂任务上表现出出色的性能。然而，之前的研究表明，这些模型通常会由于过度推理而产生显着的计算成本，例如推理轨迹之间的频繁切换（例如，思考不足）或对简单问题进行重复推理（例如，思考过度）。在这项工作中，我们暴露了一个新的威胁：对抗性输入可以被精心制作，以利用过度的推理行为，并在不影响模型效用的情况下大幅增加计算开销。因此，我们提出了一个新的损失框架，包括三个组成部分：（1）优先级交叉熵损失，通过利用LM的自回归性质强调关键令牌的标准交叉熵目标的修改;（2）过度推理损失，鼓励模型在推理过程中启动额外的推理路径;以及（3）延迟终止损失，其被设计为扩展推理过程并延迟最终输出的生成。我们针对DeepSeek-R1-Distill-LLaMA和DeepSeek-R1-Distill-Qwen上的GSM 8 K和ORCA数据集优化和评估了我们的攻击。经验结果表明，推理长度增加了3倍到9倍，而效用性能相当。此外，我们精心设计的对抗性输入具有可移植性，从而在o3-mini、o 1-mini、DeepSeek-R1和QWQ模型中引发计算负担。



## **7. LLM-Powered Intent-Based Categorization of Phishing Emails**

LLM支持的网络钓鱼电子邮件基于意图的分类 cs.CR

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14337v1) [paper-pdf](http://arxiv.org/pdf/2506.14337v1)

**Authors**: Even Eilertsen, Vasileios Mavroeidis, Gudmund Grov

**Abstract**: Phishing attacks remain a significant threat to modern cybersecurity, as they successfully deceive both humans and the defense mechanisms intended to protect them. Traditional detection systems primarily focus on email metadata that users cannot see in their inboxes. Additionally, these systems struggle with phishing emails, which experienced users can often identify empirically by the text alone. This paper investigates the practical potential of Large Language Models (LLMs) to detect these emails by focusing on their intent. In addition to the binary classification of phishing emails, the paper introduces an intent-type taxonomy, which is operationalized by the LLMs to classify emails into distinct categories and, therefore, generate actionable threat information. To facilitate our work, we have curated publicly available datasets into a custom dataset containing a mix of legitimate and phishing emails. Our results demonstrate that existing LLMs are capable of detecting and categorizing phishing emails, underscoring their potential in this domain.

摘要: 网络钓鱼攻击仍然是对现代网络安全的重大威胁，因为它们成功地欺骗了人类和旨在保护人类的防御机制。传统的检测系统主要关注用户在收件箱中看不到的电子邮件元数据。此外，这些系统还与网络钓鱼电子邮件作斗争，经验丰富的用户通常可以仅通过文本凭经验识别这些电子邮件。本文研究了大型语言模型（LLM）通过关注其意图来检测这些电子邮件的实际潜力。除了钓鱼电子邮件的二进制分类，本文还介绍了一种意图类型分类法，该分类法由LLM操作，将电子邮件分类为不同的类别，从而生成可操作的威胁信息。为了方便我们的工作，我们将公开可用的数据集整理成一个自定义数据集，其中包含合法和钓鱼电子邮件的混合。我们的研究结果表明，现有的LLM能够检测和分类钓鱼电子邮件，强调他们在这一领域的潜力。



## **8. RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?**

RL混淆：语言模型能否学会规避潜在空间预设？ cs.LG

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14261v1) [paper-pdf](http://arxiv.org/pdf/2506.14261v1)

**Authors**: Rohan Gupta, Erik Jenner

**Abstract**: Latent-space monitors aim to detect undesirable behaviours in large language models by leveraging internal model representations rather than relying solely on black-box outputs. These methods have shown promise in identifying behaviours such as deception and unsafe completions, but a critical open question remains: can LLMs learn to evade such monitors? To study this, we introduce RL-Obfuscation, in which LLMs are finetuned via reinforcement learning to bypass latent-space monitors while maintaining coherent generations. We apply RL-Obfuscation to LLMs ranging from 7B to 14B parameters and evaluate evasion success against a suite of monitors. We find that token-level latent-space monitors are highly vulnerable to this attack. More holistic monitors, such as max-pooling or attention-based probes, remain robust. Moreover, we show that adversarial policies trained to evade a single static monitor generalise to unseen monitors of the same type. Finally, we study how the policy learned by RL bypasses these monitors and find that the model can also learn to repurpose tokens to mean something different internally.

摘要: 潜伏空间监视器旨在通过利用内部模型表示而不是仅依赖黑匣子输出来检测大型语言模型中的不良行为。这些方法在识别欺骗和不安全完成等行为方面表现出了希望，但一个关键的悬而未决的问题仍然存在：LLM能否学会逃避此类监控？为了研究这一点，我们引入了RL模糊，其中LLM通过强化学习进行微调，以绕过潜在空间监视器，同时保持连贯的世代。我们对7 B到14 B参数的LLM应用RL模糊，并针对一套监视器评估规避成功率。我们发现，令牌级的潜在空间监视器非常容易受到这种攻击。更全面的监控器，如最大池或基于注意力的探测器，仍然很强大。此外，我们表明，对抗性的政策训练，以避免一个单一的静态监视器推广到看不见的监视器的相同类型。最后，我们研究了RL学习的策略如何绕过这些监视器，并发现该模型还可以学习重新使用令牌，以在内部表示不同的含义。



## **9. CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement**

捕获：上下文感知提示注入测试和稳健性增强 cs.CL

Accepted in ACL LLMSec Workshop 2025

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2505.12368v2) [paper-pdf](http://arxiv.org/pdf/2505.12368v2)

**Authors**: Gauri Kholkar, Ratinder Ahuja

**Abstract**: Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations. To demonstrate our framework's utility, we train CaptureGuard on our generated data. This new model drastically reduces both false negative and false positive rates on our context-aware datasets while also generalizing effectively to external benchmarks, establishing a path toward more robust and practical prompt injection defenses.

摘要: 提示注入仍然是大型语言模型的主要安全风险。然而，现有护栏模型在上下文感知环境中的功效仍然没有得到充分的探索，因为它们通常依赖于静态攻击基准。此外，他们还有过度防御的倾向。我们引入了CAPTURE，这是一种新型的上下文感知基准，通过最少的领域内示例来评估攻击检测和过度防御倾向。我们的实验表明，当前的即时注射护栏模型在对抗性情况下存在高假阴性，在良性情况下存在过多假阳性，凸显了严重的局限性。为了展示我们框架的实用性，我们在生成的数据上训练CaptureGuard。这种新模型大大降低了我们的上下文感知数据集上的假阴性和假阳性率，同时还有效地推广到外部基准，为更强大和实用的即时注入防御建立了一条道路。



## **10. Image Corruption-Inspired Membership Inference Attacks against Large Vision-Language Models**

针对大型视觉语言模型的受图像腐蚀启发的成员推断攻击 cs.CV

Preprint. 15 pages

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.12340v2) [paper-pdf](http://arxiv.org/pdf/2506.12340v2)

**Authors**: Zongyu Wu, Minhua Lin, Zhiwei Zhang, Fali Wang, Xianren Zhang, Xiang Zhang, Suhang Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated outstanding performance in many downstream tasks. However, LVLMs are trained on large-scale datasets, which can pose privacy risks if training images contain sensitive information. Therefore, it is important to detect whether an image is used to train the LVLM. Recent studies have investigated membership inference attacks (MIAs) against LVLMs, including detecting image-text pairs and single-modality content. In this work, we focus on detecting whether a target image is used to train the target LVLM. We design simple yet effective Image Corruption-Inspired Membership Inference Attacks (ICIMIA) against LLVLMs, which are inspired by LVLM's different sensitivity to image corruption for member and non-member images. We first perform an MIA method under the white-box setting, where we can obtain the embeddings of the image through the vision part of the target LVLM. The attacks are based on the embedding similarity between the image and its corrupted version. We further explore a more practical scenario where we have no knowledge about target LVLMs and we can only query the target LVLMs with an image and a question. We then conduct the attack by utilizing the output text embeddings' similarity. Experiments on existing datasets validate the effectiveness of our proposed attack methods under those two different settings.

摘要: 大型视觉语言模型（LVLM）在许多下游任务中表现出出色的性能。然而，LVLM是在大规模数据集上训练的，如果训练图像包含敏感信息，这可能会带来隐私风险。因此，检测图像是否用于训练LVLM非常重要。最近的研究调查了针对LVLM的成员资格推断攻击（MIA），包括检测图像-文本对和单模式内容。在这项工作中，我们重点检测目标图像是否用于训练目标LVLM。我们设计了简单而有效的针对LLVLM的图像腐败启发会员推断攻击（ICIMIA），其灵感来自LVLM对成员和非成员图像的图像腐败的不同敏感性。我们首先在白盒设置下执行MIA方法，通过目标LVLM的视觉部分获得图像的嵌入。这些攻击基于图像及其损坏版本之间的嵌入相似性。我们进一步探讨了一个更实际的情况下，我们没有关于目标LVLM的知识，我们只能查询目标LVLM的图像和一个问题。然后，我们进行攻击，利用输出文本嵌入的相似性。在现有数据集上的实验验证了我们提出的攻击方法在这两种不同设置下的有效性。



## **11. Mind the Inconspicuous: Revealing the Hidden Weakness in Aligned LLMs' Refusal Boundaries**

注意不显眼的：揭示对齐LLM拒绝边界中隐藏的弱点 cs.AI

published at USENIX Security 25

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2405.20653v3) [paper-pdf](http://arxiv.org/pdf/2405.20653v3)

**Authors**: Jiahao Yu, Haozheng Luo, Jerry Yao-Chieh Hu, Wenbo Guo, Han Liu, Xinyu Xing

**Abstract**: Recent advances in Large Language Models (LLMs) have led to impressive alignment where models learn to distinguish harmful from harmless queries through supervised finetuning (SFT) and reinforcement learning from human feedback (RLHF). In this paper, we reveal a subtle yet impactful weakness in these aligned models. We find that simply appending multiple end of sequence (eos) tokens can cause a phenomenon we call context segmentation, which effectively shifts both harmful and benign inputs closer to the refusal boundary in the hidden space.   Building on this observation, we propose a straightforward method to BOOST jailbreak attacks by appending eos tokens. Our systematic evaluation shows that this strategy significantly increases the attack success rate across 8 representative jailbreak techniques and 16 open-source LLMs, ranging from 2B to 72B parameters. Moreover, we develop a novel probing mechanism for commercial APIs and discover that major providers such as OpenAI, Anthropic, and Qwen do not filter eos tokens, making them similarly vulnerable. These findings highlight a hidden yet critical blind spot in existing alignment and content filtering approaches.   We call for heightened attention to eos tokens' unintended influence on model behaviors, particularly in production systems. Our work not only calls for an input-filtering based defense, but also points to new defenses that make refusal boundaries more robust and generalizable, as well as fundamental alignment techniques that can defend against context segmentation attacks.

摘要: 大型语言模型（LLM）的最新进展带来了令人印象深刻的一致性，模型通过监督微调（SFT）和来自人类反馈的强化学习（RL HF）来学习区分有害和无害的查询。在本文中，我们揭示了这些对齐模型中一个微妙但有影响力的弱点。我们发现，简单地添加多个序列结束（eos）令牌可能会导致一种我们称之为上下文分割的现象，该现象有效地将有害和良性输入移至更接近隐藏空间中的拒绝边界。   在这一观察的基础上，我们提出了一种通过添加eos令牌来BORST越狱攻击的简单方法。我们的系统评估表明，该策略显着提高了8种代表性越狱技术和16种开源LLM（参数范围从2B到72 B）的攻击成功率。此外，我们为商业API开发了一种新颖的探测机制，并发现OpenAI、Anthropic和Qwen等主要提供商不会过滤eos代币，这使得它们同样容易受到攻击。这些发现凸显了现有对齐和内容过滤方法中一个隐藏但关键的盲点。   我们呼吁高度关注EOS代币对模型行为的无意影响，特别是在生产系统中。我们的工作不仅呼吁基于输入过滤的防御，而且还指出了使拒绝边界更加稳健和可推广的新防御，以及可以防御上下文分割攻击的基本对齐技术。



## **12. Distraction is All You Need for Multimodal Large Language Model Jailbreaking**

分散注意力是多模态大型语言模型越狱所需的一切 cs.CV

CVPR 2025 highlight

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2502.10794v2) [paper-pdf](http://arxiv.org/pdf/2502.10794v2)

**Authors**: Zuopeng Yang, Jiluan Fan, Anli Yan, Erdun Gao, Xin Lin, Tao Li, Kanghua Mo, Changyu Dong

**Abstract**: Multimodal Large Language Models (MLLMs) bridge the gap between visual and textual data, enabling a range of advanced applications. However, complex internal interactions among visual elements and their alignment with text can introduce vulnerabilities, which may be exploited to bypass safety mechanisms. To address this, we analyze the relationship between image content and task and find that the complexity of subimages, rather than their content, is key. Building on this insight, we propose the Distraction Hypothesis, followed by a novel framework called Contrasting Subimage Distraction Jailbreaking (CS-DJ), to achieve jailbreaking by disrupting MLLMs alignment through multi-level distraction strategies. CS-DJ consists of two components: structured distraction, achieved through query decomposition that induces a distributional shift by fragmenting harmful prompts into sub-queries, and visual-enhanced distraction, realized by constructing contrasting subimages to disrupt the interactions among visual elements within the model. This dual strategy disperses the model's attention, reducing its ability to detect and mitigate harmful content. Extensive experiments across five representative scenarios and four popular closed-source MLLMs, including GPT-4o-mini, GPT-4o, GPT-4V, and Gemini-1.5-Flash, demonstrate that CS-DJ achieves average success rates of 52.40% for the attack success rate and 74.10% for the ensemble attack success rate. These results reveal the potential of distraction-based approaches to exploit and bypass MLLMs' defenses, offering new insights for attack strategies.

摘要: 多模式大型语言模型（MLLM）弥合了视觉和文本数据之间的差距，支持一系列高级应用程序。然而，视觉元素之间复杂的内部交互及其与文本的对齐可能会引入漏洞，这些漏洞可能会被利用来绕过安全机制。为了解决这个问题，我们分析了图像内容和任务之间的关系，发现子图像的复杂性而不是其内容才是关键。在这一见解的基础上，我们提出了分心假说，随后提出了一个名为对比子图像分心越狱（CS-DJ）的新颖框架，通过多级别分心策略扰乱MLLM对齐来实现越狱。CS-DJ由两个部分组成：结构化干扰，通过查询分解来实现，通过将有害提示分解为子查询来引发分布转变，以及视觉增强干扰，通过构建对比子图像来破坏模型内视觉元素之间的交互来实现。这种双重策略分散了模型的注意力，降低了其检测和减轻有害内容的能力。针对五种代表性场景和四种流行的闭源MLLM（包括GPT-4 o-mini、GPT-4 o、GPT-4V和Gemini-1.5-Flash）的广泛实验表明，CS-DJ的攻击成功率平均为52.40%，综合攻击成功率平均为74.10%。这些结果揭示了基于干扰的方法利用和绕过MLLM防御的潜力，为攻击策略提供了新的见解。



## **13. Evaluating Large Language Models for Phishing Detection, Self-Consistency, Faithfulness, and Explainability**

评估大型语言模型的网络钓鱼检测、自一致性、忠实性和可解释性 cs.CR

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13746v1) [paper-pdf](http://arxiv.org/pdf/2506.13746v1)

**Authors**: Shova Kuikel, Aritran Piplai, Palvi Aggarwal

**Abstract**: Phishing attacks remain one of the most prevalent and persistent cybersecurity threat with attackers continuously evolving and intensifying tactics to evade the general detection system. Despite significant advances in artificial intelligence and machine learning, faithfully reproducing the interpretable reasoning with classification and explainability that underpin phishing judgments remains challenging. Due to recent advancement in Natural Language Processing, Large Language Models (LLMs) show a promising direction and potential for improving domain specific phishing classification tasks. However, enhancing the reliability and robustness of classification models requires not only accurate predictions from LLMs but also consistent and trustworthy explanations aligning with those predictions. Therefore, a key question remains: can LLMs not only classify phishing emails accurately but also generate explanations that are reliably aligned with their predictions and internally self-consistent? To answer these questions, we have fine-tuned transformer based models, including BERT, Llama models, and Wizard, to improve domain relevance and make them more tailored to phishing specific distinctions, using Binary Sequence Classification, Contrastive Learning (CL) and Direct Preference Optimization (DPO). To that end, we examined their performance in phishing classification and explainability by applying the ConsistenCy measure based on SHAPley values (CC SHAP), which measures prediction explanation token alignment to test the model's internal faithfulness and consistency and uncover the rationale behind its predictions and reasoning. Overall, our findings show that Llama models exhibit stronger prediction explanation token alignment with higher CC SHAP scores despite lacking reliable decision making accuracy, whereas Wizard achieves better prediction accuracy but lower CC SHAP scores.

摘要: 网络钓鱼攻击仍然是最普遍、最持久的网络安全威胁之一，攻击者不断发展和强化策略来逃避通用检测系统。尽管人工智能和机器学习取得了重大进展，但忠实地复制支持网络钓鱼判断的具有分类和可解释性的可解释推理仍然具有挑战性。由于自然语言处理的最新进展，大型语言模型（LLM）在改进特定领域的网络钓鱼分类任务方面显示出了一个有前途的方向和潜力。然而，提高分类模型的可靠性和鲁棒性不仅需要LLM的准确预测，还需要与这些预测一致且值得信赖的解释。因此，一个关键问题仍然存在：LLM不仅可以准确地分类钓鱼电子邮件，还可以生成与其预测可靠一致且内部自洽的解释吗？为了回答这些问题，我们对基于Transformer的模型进行了微调，包括BERT、Llama模型和Wizard，以提高领域相关性，并使用二进制序列分类、对比学习（CL）和直接偏好优化（DPO），使它们更适合网络钓鱼的特定区别。为此，我们通过应用基于SHAPley值（CC SHAP）的ConistenCy测量来检查它们在网络钓鱼分类和可解释性方面的表现，该测量测量预测解释标记对齐度以测试模型的内部忠实性和一致性，并揭示其预测和推理背后的原理。总体而言，我们的研究结果表明，尽管缺乏可靠的决策准确性，Llama模型仍表现出更强的预测解释标记一致性，但CC SHAP分数更高，而Wizard则实现了更好的预测准确性，但CC SHAP分数更低。



## **14. Weakest Link in the Chain: Security Vulnerabilities in Advanced Reasoning Models**

链中最薄弱的环节：高级推理模型中的安全漏洞 cs.AI

Accepted to LLMSEC 2025

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13726v1) [paper-pdf](http://arxiv.org/pdf/2506.13726v1)

**Authors**: Arjun Krishna, Aaditya Rastogi, Erick Galinkin

**Abstract**: The introduction of advanced reasoning capabilities have improved the problem-solving performance of large language models, particularly on math and coding benchmarks. However, it remains unclear whether these reasoning models are more or less vulnerable to adversarial prompt attacks than their non-reasoning counterparts. In this work, we present a systematic evaluation of weaknesses in advanced reasoning models compared to similar non-reasoning models across a diverse set of prompt-based attack categories. Using experimental data, we find that on average the reasoning-augmented models are \emph{slightly more robust} than non-reasoning models (42.51\% vs 45.53\% attack success rate, lower is better). However, this overall trend masks significant category-specific differences: for certain attack types the reasoning models are substantially \emph{more vulnerable} (e.g., up to 32 percentage points worse on a tree-of-attacks prompt), while for others they are markedly \emph{more robust} (e.g., 29.8 points better on cross-site scripting injection). Our findings highlight the nuanced security implications of advanced reasoning in language models and emphasize the importance of stress-testing safety across diverse adversarial techniques.

摘要: 高级推理能力的引入提高了大型语言模型的问题解决性能，特别是在数学和编码基准方面。然而，目前尚不清楚这些推理模型是否比非推理模型更容易受到对抗性提示攻击。在这项工作中，我们对高级推理模型与不同的基于预算的攻击类别的类似非推理模型相比的弱点进行了系统评估。使用实验数据，我们发现，平均而言，推理增强的模型比非推理模型的攻击成功率略高（42.51比45.53，越低越好）。然而，这种总体趋势掩盖了显著的类别特定差异：对于某些攻击类型，推理模型基本上更脆弱（例如，在攻击树提示上差达32个百分点），而对于其他攻击树提示，它们明显更健壮（例如，29.8在跨站点脚本注入方面更好）。我们的研究结果强调了语言模型中高级推理的微妙安全影响，并强调了各种对抗技术压力测试安全性的重要性。



## **15. On the Feasibility of Fully AI-automated Vishing Attacks**

全人工智能自动Vising攻击的可行性 cs.CR

To appear in AsiaCCS 2025

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2409.13793v2) [paper-pdf](http://arxiv.org/pdf/2409.13793v2)

**Authors**: João Figueiredo, Afonso Carvalho, Daniel Castro, Daniel Gonçalves, Nuno Santos

**Abstract**: A vishing attack is a form of social engineering where attackers use phone calls to deceive individuals into disclosing sensitive information, such as personal data, financial information, or security credentials. Attackers exploit the perceived urgency and authenticity of voice communication to manipulate victims, often posing as legitimate entities like banks or tech support. Vishing is a particularly serious threat as it bypasses security controls designed to protect information. In this work, we study the potential for vishing attacks to escalate with the advent of AI. In theory, AI-powered software bots may have the ability to automate these attacks by initiating conversations with potential victims via phone calls and deceiving them into disclosing sensitive information. To validate this thesis, we introduce ViKing, an AI-powered vishing system developed using publicly available AI technology. It relies on a Large Language Model (LLM) as its core cognitive processor to steer conversations with victims, complemented by a pipeline of speech-to-text and text-to-speech modules that facilitate audio-text conversion in phone calls. Through a controlled social experiment involving 240 participants, we discovered that ViKing has successfully persuaded many participants to reveal sensitive information, even those who had been explicitly warned about the risk of vishing campaigns. Interactions with ViKing's bots were generally considered realistic. From these findings, we conclude that tools like ViKing may already be accessible to potential malicious actors, while also serving as an invaluable resource for cyber awareness programs.

摘要: 钓鱼攻击是一种社会工程形式，攻击者利用电话欺骗个人披露敏感信息，例如个人数据、财务信息或安全凭证。攻击者利用语音通信的紧迫性和真实性来操纵受害者，通常冒充银行或技术支持等合法实体。Vising是一个特别严重的威胁，因为它绕过了旨在保护信息的安全控制。在这项工作中，我们研究了随着人工智能的出现，钓鱼攻击升级的可能性。理论上，人工智能驱动的软件机器人可能有能力通过电话发起与潜在受害者的对话并欺骗他们披露敏感信息来自动化这些攻击。为了验证这一论文，我们引入了ViKing，这是一个使用公开的人工智能技术开发的人工智能驱动的钓鱼系统。它依赖大语言模型（LLM）作为其核心认知处理器来引导与受害者的对话，并辅之以语音到文本和文本到语音模块管道，以促进电话通话中的音频到文本转换。通过一项涉及240名参与者的受控社会实验，我们发现ViKing成功说服许多参与者透露敏感信息，即使是那些被明确警告参加活动的风险的人。与维京机器人的互动通常被认为是现实的。从这些发现中，我们得出结论，像Viking这样的工具可能已经可以被潜在的恶意行为者使用，同时也是网络意识计划的宝贵资源。



## **16. Benchmarking Practices in LLM-driven Offensive Security: Testbeds, Metrics, and Experiment Design**

法学硕士驱动的攻击性安全中的基准实践：测试床、工作组和实验设计 cs.CR

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2504.10112v2) [paper-pdf](http://arxiv.org/pdf/2504.10112v2)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: Large Language Models (LLMs) have emerged as a powerful approach for driving offensive penetration-testing tooling. Due to the opaque nature of LLMs, empirical methods are typically used to analyze their efficacy. The quality of this analysis is highly dependent on the chosen testbed, captured metrics and analysis methods employed.   This paper analyzes the methodology and benchmarking practices used for evaluating Large Language Model (LLM)-driven attacks, focusing on offensive uses of LLMs in cybersecurity. We review 19 research papers detailing 18 prototypes and their respective testbeds.   We detail our findings and provide actionable recommendations for future research, emphasizing the importance of extending existing testbeds, creating baselines, and including comprehensive metrics and qualitative analysis. We also note the distinction between security research and practice, suggesting that CTF-based challenges may not fully represent real-world penetration testing scenarios.

摘要: 大型语言模型（LLM）已成为驱动攻击性渗透测试工具的强大方法。由于LLM的不透明性质，通常使用经验方法来分析其功效。此分析的质量高度取决于所选择的测试平台、捕获的指标和所采用的分析方法。   本文分析了用于评估大型语言模型（LLM）驱动的攻击的方法论和基准实践，重点关注LLM在网络安全中的攻击性使用。我们回顾了19篇研究论文，详细介绍了18个原型及其各自的测试平台。   我们详细介绍了我们的调查结果，并为未来的研究提供了可操作的建议，强调了扩展现有测试平台、创建基线以及包括全面指标和定性分析的重要性。我们还注意到安全研究和实践之间的区别，这表明基于CTF的挑战可能无法完全代表现实世界的渗透测试场景。



## **17. From Promise to Peril: Rethinking Cybersecurity Red and Blue Teaming in the Age of LLMs**

从承诺到危险：重新思考法学硕士时代的网络安全红蓝合作 cs.CR

10 pages

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13434v1) [paper-pdf](http://arxiv.org/pdf/2506.13434v1)

**Authors**: Alsharif Abuadbba, Chris Hicks, Kristen Moore, Vasilios Mavroudis, Burak Hasircioglu, Diksha Goel, Piers Jennings

**Abstract**: Large Language Models (LLMs) are set to reshape cybersecurity by augmenting red and blue team operations. Red teams can exploit LLMs to plan attacks, craft phishing content, simulate adversaries, and generate exploit code. Conversely, blue teams may deploy them for threat intelligence synthesis, root cause analysis, and streamlined documentation. This dual capability introduces both transformative potential and serious risks.   This position paper maps LLM applications across cybersecurity frameworks such as MITRE ATT&CK and the NIST Cybersecurity Framework (CSF), offering a structured view of their current utility and limitations. While LLMs demonstrate fluency and versatility across various tasks, they remain fragile in high-stakes, context-heavy environments. Key limitations include hallucinations, limited context retention, poor reasoning, and sensitivity to prompts, which undermine their reliability in operational settings.   Moreover, real-world integration raises concerns around dual-use risks, adversarial misuse, and diminished human oversight. Malicious actors could exploit LLMs to automate reconnaissance, obscure attack vectors, and lower the technical threshold for executing sophisticated attacks.   To ensure safer adoption, we recommend maintaining human-in-the-loop oversight, enhancing model explainability, integrating privacy-preserving mechanisms, and building systems robust to adversarial exploitation. As organizations increasingly adopt AI driven cybersecurity, a nuanced understanding of LLMs' risks and operational impacts is critical to securing their defensive value while mitigating unintended consequences.

摘要: 大型语言模型（LLM）将通过增强红色和蓝色团队运营来重塑网络安全。红色团队可以利用LLM来计划攻击、制作网络钓鱼内容、模拟对手并生成利用代码。相反，蓝色团队可能会部署它们来进行威胁情报合成、根本原因分析和简化文档。这种双重能力既带来了变革潜力，也带来了严重的风险。   这份立场文件绘制了跨MITRE ATT & CK和NIH网络安全框架（CSF）等网络安全框架的LLM应用程序，提供了其当前效用和局限性的结构化视图。虽然LLM在各种任务中表现出流畅性和多功能性，但它们在高风险、上下文密集的环境中仍然很脆弱。主要局限性包括幻觉、上下文保留有限、推理能力差以及对提示的敏感性，这些都削弱了其在操作环境中的可靠性。   此外，现实世界的一体化引发了人们对两用风险、对抗性滥用和人类监督减少的担忧。恶意行为者可以利用LLM来自动侦察、掩盖攻击载体并降低执行复杂攻击的技术门槛。   为了确保更安全的采用，我们建议保持人在循环中的监督、增强模型的可解释性、集成隐私保护机制并构建对对抗性剥削稳健的系统。随着组织越来越多地采用人工智能驱动的网络安全，对LLM的风险和运营影响的细致了解对于确保其防御价值同时减轻意外后果至关重要。



## **18. Mitigating Safety Fallback in Editing-based Backdoor Injection on LLMs**

减轻LLM上基于编辑的后门注入中的安全回退 cs.CL

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13285v1) [paper-pdf](http://arxiv.org/pdf/2506.13285v1)

**Authors**: Houcheng Jiang, Zetong Zhao, Junfeng Fang, Haokai Ma, Ruipeng Wang, Yang Deng, Xiang Wang, Xiangnan He

**Abstract**: Large language models (LLMs) have shown strong performance across natural language tasks, but remain vulnerable to backdoor attacks. Recent model editing-based approaches enable efficient backdoor injection by directly modifying parameters to map specific triggers to attacker-desired responses. However, these methods often suffer from safety fallback, where the model initially responds affirmatively but later reverts to refusals due to safety alignment. In this work, we propose DualEdit, a dual-objective model editing framework that jointly promotes affirmative outputs and suppresses refusal responses. To address two key challenges -- balancing the trade-off between affirmative promotion and refusal suppression, and handling the diversity of refusal expressions -- DualEdit introduces two complementary techniques. (1) Dynamic loss weighting calibrates the objective scale based on the pre-edited model to stabilize optimization. (2) Refusal value anchoring compresses the suppression target space by clustering representative refusal value vectors, reducing optimization conflict from overly diverse token sets. Experiments on safety-aligned LLMs show that DualEdit improves attack success by 9.98\% and reduces safety fallback rate by 10.88\% over baselines.

摘要: 大型语言模型（LLM）在自然语言任务中表现出了强劲的性能，但仍然容易受到后门攻击。最近基于模型编辑的方法通过直接修改参数将特定触发器映射到攻击者期望的响应来实现高效的后门注入。然而，这些方法通常会受到安全倒退的影响，模型最初会做出肯定的反应，但后来由于安全调整而恢复拒绝。在这项工作中，我们提出了DualEdit，这是一个双目标模型编辑框架，可以共同促进肯定输出并抑制拒绝回应。为了解决两个关键挑战--平衡肯定促进和拒绝抑制之间的权衡，以及处理拒绝表达的多样性-- DualEdit引入了两种补充技术。(1)动态损失加权基于预先编辑的模型校准目标规模，稳定优化。(2)拒绝值锚定通过对代表性拒绝值载体进行聚集来压缩抑制目标空间，从而减少过度多样化的令牌集带来的优化冲突。在安全一致的LLM上的实验表明，DualEdit将攻击成功率提高了9.98%，并将安全回退率降低了10.88%。



## **19. Navigating the Black Box: Leveraging LLMs for Effective Text-Level Graph Injection Attacks**

导航黑匣子：利用LLM进行有效的文本级图形注入攻击 cs.AI

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13276v1) [paper-pdf](http://arxiv.org/pdf/2506.13276v1)

**Authors**: Yuefei Lyu, Chaozhuo Li, Xi Zhang, Tianle Zhang

**Abstract**: Text-attributed graphs (TAGs) integrate textual data with graph structures, providing valuable insights in applications such as social network analysis and recommendation systems. Graph Neural Networks (GNNs) effectively capture both topological structure and textual information in TAGs but are vulnerable to adversarial attacks. Existing graph injection attack (GIA) methods assume that attackers can directly manipulate the embedding layer, producing non-explainable node embeddings. Furthermore, the effectiveness of these attacks often relies on surrogate models with high training costs. Thus, this paper introduces ATAG-LLM, a novel black-box GIA framework tailored for TAGs. Our approach leverages large language models (LLMs) to generate interpretable text-level node attributes directly, ensuring attacks remain feasible in real-world scenarios. We design strategies for LLM prompting that balance exploration and reliability to guide text generation, and propose a similarity assessment method to evaluate attack text effectiveness in disrupting graph homophily. This method efficiently perturbs the target node with minimal training costs in a strict black-box setting, ensuring a text-level graph injection attack for TAGs. Experiments on real-world TAG datasets validate the superior performance of ATAG-LLM compared to state-of-the-art embedding-level and text-level attack methods.

摘要: 文本属性图（TAG）将文本数据与图结构集成，为社交网络分析和推荐系统等应用程序提供有价值的见解。图形神经网络（GNN）有效捕获TAG中的拓扑结构和文本信息，但很容易受到对抗攻击。现有的图注入攻击（GIA）方法假设攻击者可以直接操纵嵌入层，从而产生不可解释的节点嵌入。此外，这些攻击的有效性通常依赖于具有高训练成本的代理模型。因此，本文介绍了ATAG-LLM，这是一种为TAG量身定制的新型黑匣子GIA框架。我们的方法利用大型语言模型（LLM）直接生成可解释的文本级节点属性，确保攻击在现实世界场景中仍然可行。我们为LLM设计了策略，促使平衡探索性和可靠性来指导文本生成，并提出了一种相似性评估方法来评估攻击文本破坏图同质性的有效性。该方法在严格的黑匣子设置中以最小的训练成本有效地扰乱目标节点，确保对TAG的文本级图注入攻击。现实世界TAG数据集上的实验验证了ATAG-LLM与最先进的嵌入级和文本级攻击方法相比的优越性能。



## **20. Detecting Hard-Coded Credentials in Software Repositories via LLMs**

通过LLM检测软件存储库中的硬编码凭据 cs.CR

Accepted to the ACM Digital Threats: Research and Practice (DTRAP)

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13090v1) [paper-pdf](http://arxiv.org/pdf/2506.13090v1)

**Authors**: Chidera Biringa, Gokhan Kul

**Abstract**: Software developers frequently hard-code credentials such as passwords, generic secrets, private keys, and generic tokens in software repositories, even though it is strictly advised against due to the severe threat to the security of the software. These credentials create attack surfaces exploitable by a potential adversary to conduct malicious exploits such as backdoor attacks. Recent detection efforts utilize embedding models to vectorize textual credentials before passing them to classifiers for predictions. However, these models struggle to discriminate between credentials with contextual and complex sequences resulting in high false positive predictions. Context-dependent Pre-trained Language Models (PLMs) or Large Language Models (LLMs) such as Generative Pre-trained Transformers (GPT) tackled this drawback by leveraging the transformer neural architecture capacity for self-attention to capture contextual dependencies between words in input sequences. As a result, GPT has achieved wide success in several natural language understanding endeavors. Hence, we assess LLMs to represent these observations and feed extracted embedding vectors to a deep learning classifier to detect hard-coded credentials. Our model outperforms the current state-of-the-art by 13% in F1 measure on the benchmark dataset. We have made all source code and data publicly available to facilitate the reproduction of all results presented in this paper.

摘要: 软件开发人员经常在软件存储库中硬编码凭证，例如密码、通用秘密、私有密钥和通用令牌，尽管由于对软件安全性的严重威胁而被严格建议不要这样做。这些凭证创建了可被潜在对手利用的攻击表面，以实施后门攻击等恶意利用。最近的检测工作利用嵌入模型来对文本凭证进行载体化，然后将其传递给分类器进行预测。然而，这些模型很难区分具有上下文和复杂序列的证书，从而导致高假阳性预测。上下文相关的预训练语言模型（PLM）或大型语言模型（LLM），如生成式预训练转换器（GPT），通过利用Transformer神经架构的自我注意能力来捕获输入序列中单词之间的上下文依赖关系，解决了这个缺点。因此，GPT在多项自然语言理解工作中取得了广泛成功。因此，我们评估LLM来表示这些观察，并将提取的嵌入载体提供给深度学习分类器以检测硬编码凭证。在基准数据集的F1指标中，我们的模型比当前最先进的模型高出13%。我们已公开所有源代码和数据，以促进本文中提出的所有结果的复制。



## **21. Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions**

轻松说话：通过简单的互动引发法学硕士的有害越狱 cs.LG

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2502.04322v2) [paper-pdf](http://arxiv.org/pdf/2502.04322v2)

**Authors**: Yik Siu Chan, Narutatsu Ri, Yuxin Xiao, Marzyeh Ghassemi

**Abstract**: Despite extensive safety alignment efforts, large language models (LLMs) remain vulnerable to jailbreak attacks that elicit harmful behavior. While existing studies predominantly focus on attack methods that require technical expertise, two critical questions remain underexplored: (1) Are jailbroken responses truly useful in enabling average users to carry out harmful actions? (2) Do safety vulnerabilities exist in more common, simple human-LLM interactions? In this paper, we demonstrate that LLM responses most effectively facilitate harmful actions when they are both actionable and informative--two attributes easily elicited in multi-step, multilingual interactions. Using this insight, we propose HarmScore, a jailbreak metric that measures how effectively an LLM response enables harmful actions, and Speak Easy, a simple multi-step, multilingual attack framework. Notably, by incorporating Speak Easy into direct request and jailbreak baselines, we see an average absolute increase of 0.319 in Attack Success Rate and 0.426 in HarmScore in both open-source and proprietary LLMs across four safety benchmarks. Our work reveals a critical yet often overlooked vulnerability: Malicious users can easily exploit common interaction patterns for harmful intentions.

摘要: 尽管做出了广泛的安全调整工作，大型语言模型（LLM）仍然容易受到引发有害行为的越狱攻击。虽然现有的研究主要集中在需要技术专业知识的攻击方法上，但有两个关键问题仍未得到充分研究：（1）越狱响应是否真的有助于让普通用户实施有害行为？(2)更常见、简单的人类与LLM互动中是否存在安全漏洞？在本文中，我们证明，当LLM响应既可操作又提供信息时，它们最有效地促进了有害行为--这两个属性在多步骤、多语言交互中很容易被引出。利用这一见解，我们提出了HarmScore，这是一种越狱指标，衡量LLM响应实施有害行为的有效性，还提出了Speak Easy，这是一种简单的多步骤、多语言攻击框架。值得注意的是，通过将Speak Easy纳入直接请求和越狱基线，我们发现开源和专有LLM在四个安全基准上的攻击成功率平均绝对增加了0.319，HarmScore平均绝对增加了0.426。我们的工作揭示了一个关键但经常被忽视的漏洞：恶意用户可以很容易地利用常见的交互模式来实现有害意图。



## **22. Universal Jailbreak Suffixes Are Strong Attention Hijackers**

通用越狱后缀是强烈的注意力劫持者 cs.CR

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12880v1) [paper-pdf](http://arxiv.org/pdf/2506.12880v1)

**Authors**: Matan Ben-Tov, Mor Geva, Mahmood Sharif

**Abstract**: We study suffix-based jailbreaks$\unicode{x2013}$a powerful family of attacks against large language models (LLMs) that optimize adversarial suffixes to circumvent safety alignment. Focusing on the widely used foundational GCG attack (Zou et al., 2023), we observe that suffixes vary in efficacy: some markedly more universal$\unicode{x2013}$generalizing to many unseen harmful instructions$\unicode{x2013}$than others. We first show that GCG's effectiveness is driven by a shallow, critical mechanism, built on the information flow from the adversarial suffix to the final chat template tokens before generation. Quantifying the dominance of this mechanism during generation, we find GCG irregularly and aggressively hijacks the contextualization process. Crucially, we tie hijacking to the universality phenomenon, with more universal suffixes being stronger hijackers. Subsequently, we show that these insights have practical implications: GCG universality can be efficiently enhanced (up to $\times$5 in some cases) at no additional computational cost, and can also be surgically mitigated, at least halving attack success with minimal utility loss. We release our code and data at http://github.com/matanbt/interp-jailbreak.

摘要: 我们研究基于后缀的越狱$\unicode{x2013}$这是一个针对大型语言模型（LLM）的强大攻击家族，这些模型优化对抗性后缀以规避安全对齐。专注于广泛使用的基础GCG攻击（Zou等人，2023），我们观察到后缀的功效各不相同：有些后缀明显更通用的$\unicode{x2013}$一般化为许多不可见的有害指令$\unicode{x2013}$。我们首先表明，GCG的有效性是由一种肤浅的关键机制驱动的，该机制建立在从对抗性后缀到生成之前的最终聊天模板令牌的信息流之上。量化这种机制在生成过程中的主导地位，我们发现GCG不规则且积极地劫持了情境化过程。至关重要的是，我们将劫持与普遍现象联系起来，更普遍的后缀意味着更强大的劫持者。随后，我们证明了这些见解具有实际意义：GCG通用性可以在不需要额外计算成本的情况下有效增强（在某些情况下高达5美元），并且还可以通过手术来减轻，至少将攻击成功率减半，并将效用损失最小。我们在http://github.com/matanbt/interp-jailbreak上发布我们的代码和数据。



## **23. Transforming Chatbot Text: A Sequence-to-Sequence Approach**

改造聊天机器人文本：序列到序列的方法 cs.CL

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12843v1) [paper-pdf](http://arxiv.org/pdf/2506.12843v1)

**Authors**: Natesh Reddy, Mark Stamp

**Abstract**: Due to advances in Large Language Models (LLMs) such as ChatGPT, the boundary between human-written text and AI-generated text has become blurred. Nevertheless, recent work has demonstrated that it is possible to reliably detect GPT-generated text. In this paper, we adopt a novel strategy to adversarially transform GPT-generated text using sequence-to-sequence (Seq2Seq) models, with the goal of making the text more human-like. We experiment with the Seq2Seq models T5-small and BART which serve to modify GPT-generated sentences to include linguistic, structural, and semantic components that may be more typical of human-authored text. Experiments show that classification models trained to distinguish GPT-generated text are significantly less accurate when tested on text that has been modified by these Seq2Seq models. However, after retraining classification models on data generated by our Seq2Seq technique, the models are able to distinguish the transformed GPT-generated text from human-generated text with high accuracy. This work adds to the accumulating knowledge of text transformation as a tool for both attack -- in the sense of defeating classification models -- and defense -- in the sense of improved classifiers -- thereby advancing our understanding of AI-generated text.

摘要: 由于ChatGPT等大型语言模型（LLM）的进步，人类书面文本和人工智能生成文本之间的界限变得模糊。尽管如此，最近的工作表明，可以可靠地检测GPT生成的文本。在本文中，我们采用了一种新颖的策略，使用序列到序列（Seq 2Seq）模型对GPT生成的文本进行反向转换，目标是使文本更像人。我们实验了Seq 2Seq模型T5-small和BART，这些模型用于修改GPT生成的句子，以包括可能更典型的人类创作文本的语言、结构和语义成分。实验表明，当对已由这些Seq 2Seq模型修改的文本进行测试时，为区分GPT生成的文本而训练的分类模型的准确性明显较差。然而，在对我们的Seq 2Seq技术生成的数据重新训练分类模型后，这些模型能够高准确性区分转换后的GPT生成文本与人类生成文本。这项工作增加了文本转换作为攻击工具（从击败分类模型的意义上来说）和防御（从改进分类器的意义上来说）的知识的积累，从而促进我们对人工智能生成的文本的理解。



## **24. I Know What You Said: Unveiling Hardware Cache Side-Channels in Local Large Language Model Inference**

我知道你说什么：揭开本地大型语言模型推理中的硬件缓存侧通道 cs.CR

Submitted for review in January 22, 2025, revised under shepherding

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2505.06738v3) [paper-pdf](http://arxiv.org/pdf/2505.06738v3)

**Authors**: Zibo Gao, Junjie Hu, Feng Guo, Yixin Zhang, Yinglong Han, Siyuan Liu, Haiyang Li, Zhiqiang Lv

**Abstract**: Large Language Models (LLMs) that can be deployed locally have recently gained popularity for privacy-sensitive tasks, with companies such as Meta, Google, and Intel playing significant roles in their development. However, the security of local LLMs through the lens of hardware cache side-channels remains unexplored. In this paper, we unveil novel side-channel vulnerabilities in local LLM inference: token value and token position leakage, which can expose both the victim's input and output text, thereby compromising user privacy. Specifically, we found that adversaries can infer the token values from the cache access patterns of the token embedding operation, and deduce the token positions from the timing of autoregressive decoding phases. To demonstrate the potential of these leaks, we design a novel eavesdropping attack framework targeting both open-source and proprietary LLM inference systems. The attack framework does not directly interact with the victim's LLM and can be executed without privilege.   We evaluate the attack on a range of practical local LLM deployments (e.g., Llama, Falcon, and Gemma), and the results show that our attack achieves promising accuracy. The restored output and input text have an average edit distance of 5.2% and 17.3% to the ground truth, respectively. Furthermore, the reconstructed texts achieve average cosine similarity scores of 98.7% (input) and 98.0% (output).

摘要: 可以在本地部署的大型语言模型（LLM）最近在隐私敏感任务中越来越受欢迎，Meta、谷歌和英特尔等公司在其开发中发挥了重要作用。然而，通过硬件缓存侧通道的视角来探讨本地LLM的安全性仍然有待探索。在本文中，我们揭示了本地LLM推断中的新型侧通道漏洞：令牌值和令牌位置泄露，它可以暴露受害者的输入和输出文本，从而损害用户隐私。具体来说，我们发现对手可以从令牌嵌入操作的缓存访问模式中推断令牌值，并从自回归解码阶段的时间推断令牌位置。为了证明这些泄漏的潜力，我们设计了一个新的窃听攻击框架，针对开源和专有的LLM推理系统。攻击框架不直接与受害者的LLM交互，并且可以在没有特权的情况下执行。   我们评估了对一系列实际本地LLM部署的攻击（例如，Llama、Falcon和Gemma），结果表明我们的攻击达到了令人满意的准确性。恢复的输出和输入文本与地面真相的平均编辑距离分别为5.2%和17.3%。此外，重建的文本的平均cos相似度评分为98.7%（输入）和98.0%（输出）。



## **25. Can We Infer Confidential Properties of Training Data from LLMs?**

我们可以从LLM推断培训数据的机密属性吗？ cs.LG

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.10364v2) [paper-pdf](http://arxiv.org/pdf/2506.10364v2)

**Authors**: Pengrun Huang, Chhavi Yadav, Ruihan Wu, Kamalika Chaudhuri

**Abstract**: Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs.

摘要: 大型语言模型（LLM）越来越多地针对特定领域的数据集进行微调，以支持医疗保健、金融和法律等领域的应用。这些微调数据集通常具有敏感且保密的厕所级属性（例如患者人口统计数据或疾病患病率），这些属性不打算被披露。虽然之前的工作研究了对区分模型的属性推断攻击（例如，图像分类模型）和生成模型（例如，图像数据的GAN），目前尚不清楚此类攻击是否转移到LLM。在这项工作中，我们引入了PropInfer，这是一项基准任务，用于在两种微调范式下评估LLM中的属性推理：问答和聊天完成。我们的基准基于ChatDoctor数据集构建，包括一系列属性类型和任务配置。我们进一步提出了两种定制攻击：基于预算的生成攻击和利用词频信号的影子模型攻击。对多个预训练的LLM进行的经验评估显示了我们的攻击的成功，揭示了LLM中以前未被识别的漏洞。



## **26. SecurityLingua: Efficient Defense of LLM Jailbreak Attacks via Security-Aware Prompt Compression**

SecureLingua：通过安全意识提示压缩有效防御LLM越狱攻击 cs.CR

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12707v1) [paper-pdf](http://arxiv.org/pdf/2506.12707v1)

**Authors**: Yucheng Li, Surin Ahn, Huiqiang Jiang, Amir H. Abdi, Yuqing Yang, Lili Qiu

**Abstract**: Large language models (LLMs) have achieved widespread adoption across numerous applications. However, many LLMs are vulnerable to malicious attacks even after safety alignment. These attacks typically bypass LLMs' safety guardrails by wrapping the original malicious instructions inside adversarial jailbreaks prompts. Previous research has proposed methods such as adversarial training and prompt rephrasing to mitigate these safety vulnerabilities, but these methods often reduce the utility of LLMs or lead to significant computational overhead and online latency. In this paper, we propose SecurityLingua, an effective and efficient approach to defend LLMs against jailbreak attacks via security-oriented prompt compression. Specifically, we train a prompt compressor designed to discern the "true intention" of the input prompt, with a particular focus on detecting the malicious intentions of adversarial prompts. Then, in addition to the original prompt, the intention is passed via the system prompt to the target LLM to help it identify the true intention of the request. SecurityLingua ensures a consistent user experience by leaving the original input prompt intact while revealing the user's potentially malicious intention and stimulating the built-in safety guardrails of the LLM. Moreover, thanks to prompt compression, SecurityLingua incurs only a negligible overhead and extra token cost compared to all existing defense methods, making it an especially practical solution for LLM defense. Experimental results demonstrate that SecurityLingua can effectively defend against malicious attacks and maintain utility of the LLM with negligible compute and latency overhead. Our code is available at https://aka.ms/SecurityLingua.

摘要: 大型语言模型（LLM）已经在许多应用程序中得到了广泛采用。然而，许多LLM即使在安全对齐之后也容易受到恶意攻击。这些攻击通常通过将原始恶意指令包装在对抗性越狱提示中来绕过LLM的安全护栏。以前的研究已经提出了对抗性训练和快速改写等方法来减轻这些安全漏洞，但这些方法通常会降低LLM的实用性或导致显着的计算开销和在线延迟。在本文中，我们提出了SecureLingua，这是一种通过面向安全的即时压缩来保护LLM免受越狱攻击的有效且高效的方法。具体来说，我们训练了一个提示压缩器，旨在识别输入提示的“真实意图”，特别关注检测对抗提示的恶意意图。然后，除了原始提示之外，意图还通过系统提示传递给目标LLM，以帮助其识别请求的真实意图。SecureLingua通过保持原始输入提示不变，同时揭露用户潜在的恶意意图并刺激LLM的内置安全护栏，确保一致的用户体验。此外，由于快速压缩，与所有现有防御方法相比，SecureLingua仅产生可忽略不计的费用和额外的令牌成本，使其成为LLM防御特别实用的解决方案。实验结果表明，SecureLingua可以有效防御恶意攻击并保持LLM的实用性，而计算和延迟负担可以忽略不计。我们的代码可在https://aka.ms/SecurityLingua上获取。



## **27. Alphabet Index Mapping: Jailbreaking LLMs through Semantic Dissimilarity**

字母索引映射：通过语义差异越狱LLM cs.CR

10 pages, 2 figures, 3 tables

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12685v1) [paper-pdf](http://arxiv.org/pdf/2506.12685v1)

**Authors**: Bilal Saleh Husain

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their susceptibility to adversarial attacks, particularly jailbreaking, poses significant safety and ethical concerns. While numerous jailbreak methods exist, many suffer from computational expense, high token usage, or complex decoding schemes. Liu et al. (2024) introduced FlipAttack, a black-box method that achieves high attack success rates (ASR) through simple prompt manipulation. This paper investigates the underlying mechanisms of FlipAttack's effectiveness by analyzing the semantic changes induced by its flipping modes. We hypothesize that semantic dissimilarity between original and manipulated prompts is inversely correlated with ASR. To test this, we examine embedding space visualizations (UMAP, KDE) and cosine similarities for FlipAttack's modes. Furthermore, we introduce a novel adversarial attack, Alphabet Index Mapping (AIM), designed to maximize semantic dissimilarity while maintaining simple decodability. Experiments on GPT-4 using a subset of AdvBench show AIM and its variant AIM+FWO achieve a 94% ASR, outperforming FlipAttack and other methods on this subset. Our findings suggest that while high semantic dissimilarity is crucial, a balance with decoding simplicity is key for successful jailbreaking. This work contributes to a deeper understanding of adversarial prompt mechanics and offers a new, effective jailbreak technique.

摘要: 大型语言模型（LLM）已表现出非凡的能力，但它们对对抗攻击（尤其是越狱）的敏感性带来了重大的安全和道德问题。虽然存在多种越狱方法，但许多方法都面临计算成本、令牌使用率高或解码方案复杂的问题。Liu等人（2024）介绍了FlipAttack，这是一种黑匣子方法，通过简单的提示操作来实现高攻击成功率（ASB）。本文通过分析FlipAttack翻转模式引发的语义变化，探讨了FlipAttack有效性的潜在机制。我们假设原始提示和操纵提示之间的语义差异与ASB呈负相关。为了测试这一点，我们检查了FlipAttack模式的嵌入空间可视化（UMAP、TEK）和cos相似性。此外，我们还引入了一种新型的对抗攻击，即字母索引映射（AIM），旨在最大化语义差异，同时保持简单的可解码性。使用AdvBench的一个子集对GPT-4进行的实验表明，AIM及其变体AIM+ FBO实现了94%的ASB，优于FlipAttack和该子集的其他方法。我们的研究结果表明，虽然高度的语义差异至关重要，但与解码简单性的平衡是成功越狱的关键。这项工作有助于更深入地理解对抗提示机制，并提供了一种新的、有效的越狱技术。



## **28. MEraser: An Effective Fingerprint Erasure Approach for Large Language Models**

MEraser：大型语言模型的有效指纹擦除方法 cs.CR

Accepted by ACL 2025, Main Conference, Long Paper

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12551v1) [paper-pdf](http://arxiv.org/pdf/2506.12551v1)

**Authors**: Jingxuan Zhang, Zhenhua Xu, Rui Hu, Wenpeng Xing, Xuhong Zhang, Meng Han

**Abstract**: Large Language Models (LLMs) have become increasingly prevalent across various sectors, raising critical concerns about model ownership and intellectual property protection. Although backdoor-based fingerprinting has emerged as a promising solution for model authentication, effective attacks for removing these fingerprints remain largely unexplored. Therefore, we present Mismatched Eraser (MEraser), a novel method for effectively removing backdoor-based fingerprints from LLMs while maintaining model performance. Our approach leverages a two-phase fine-tuning strategy utilizing carefully constructed mismatched and clean datasets. Through extensive evaluation across multiple LLM architectures and fingerprinting methods, we demonstrate that MEraser achieves complete fingerprinting removal while maintaining model performance with minimal training data of fewer than 1,000 samples. Furthermore, we introduce a transferable erasure mechanism that enables effective fingerprinting removal across different models without repeated training. In conclusion, our approach provides a practical solution for fingerprinting removal in LLMs, reveals critical vulnerabilities in current fingerprinting techniques, and establishes comprehensive evaluation benchmarks for developing more resilient model protection methods in the future.

摘要: 大型语言模型（LLM）在各个领域变得越来越普遍，引发了人们对模型所有权和知识产权保护的严重担忧。尽管基于后门的指纹识别已成为模型认证的一种有希望的解决方案，但用于删除这些指纹的有效攻击在很大程度上仍然没有被探索。因此，我们提出了Mmatched Eraser（MEraser），这是一种新型方法，可以有效地从LLM中删除基于后门的指纹，同时保持模型性能。我们的方法利用两阶段微调策略，利用精心构建的不匹配且干净的数据集。通过对多种LLM架构和指纹识别方法的广泛评估，我们证明MEraser可以通过少于1，000个样本的最少训练数据实现完全指纹识别，同时保持模型性能。此外，我们引入了一种可转移的擦除机制，可以在不同模型之间有效地去除指纹，而无需重复训练。总之，我们的方法为LLM中的指纹删除提供了一种实用的解决方案，揭示了当前指纹技术中的关键漏洞，并为未来开发更具弹性的模型保护方法建立了全面的评估基准。



## **29. Pushing the Limits of Safety: A Technical Report on the ATLAS Challenge 2025**

突破安全极限：2025年ATLAS挑战赛技术报告 cs.CR

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12430v1) [paper-pdf](http://arxiv.org/pdf/2506.12430v1)

**Authors**: Zonghao Ying, Siyang Wu, Run Hao, Peng Ying, Shixuan Sun, Pengyu Chen, Junze Chen, Hao Du, Kaiwen Shen, Shangkun Wu, Jiwei Wei, Shiyuan He, Yang Yang, Xiaohai Xu, Ke Ma, Qianqian Xu, Qingming Huang, Shi Lin, Xun Wang, Changting Lin, Meng Han, Yilei Jiang, Siqi Lai, Yaozhi Zheng, Yifei Song, Xiangyu Yue, Zonglei Jing, Tianyuan Zhang, Zhilei Zhu, Aishan Liu, Jiakai Wang, Siyuan Liang, Xianglong Kong, Hainan Li, Junjie Mu, Haotong Qin, Yue Yu, Lei Chen, Felix Juefei-Xu, Qing Guo, Xinyun Chen, Yew Soon Ong, Xianglong Liu, Dawn Song, Alan Yuille, Philip Torr, Dacheng Tao

**Abstract**: Multimodal Large Language Models (MLLMs) have enabled transformative advancements across diverse applications but remain susceptible to safety threats, especially jailbreak attacks that induce harmful outputs. To systematically evaluate and improve their safety, we organized the Adversarial Testing & Large-model Alignment Safety Grand Challenge (ATLAS) 2025}. This technical report presents findings from the competition, which involved 86 teams testing MLLM vulnerabilities via adversarial image-text attacks in two phases: white-box and black-box evaluations. The competition results highlight ongoing challenges in securing MLLMs and provide valuable guidance for developing stronger defense mechanisms. The challenge establishes new benchmarks for MLLM safety evaluation and lays groundwork for advancing safer multimodal AI systems. The code and data for this challenge are openly available at https://github.com/NY1024/ATLAS_Challenge_2025.

摘要: 多模式大型语言模型（MLLM）在不同的应用程序中实现了变革性的进步，但仍然容易受到安全威胁，尤其是引发有害输出的越狱攻击。为了系统地评估和提高其安全性，我们组织了对抗性测试和大模型对齐安全大挑战赛（ATLAS）2025。本技术报告介绍了比赛的结果，其中86个团队通过对抗性图像文本攻击分两个阶段测试MLLM漏洞：白盒和黑匣子评估。竞赛结果凸显了确保MLLM方面持续存在的挑战，并为开发更强大的防御机制提供了宝贵的指导。该挑战为MLLM安全评估建立了新的基准，并为推进更安全的多模式人工智能系统奠定了基础。此挑战的代码和数据可在https://github.com/NY1024/ATLAS_Challenge_2025上公开获取。



## **30. Exploring the Secondary Risks of Large Language Models**

探索大型语言模型的次要风险 cs.LG

18 pages, 5 figures

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12382v1) [paper-pdf](http://arxiv.org/pdf/2506.12382v1)

**Authors**: Jiawei Chen, Zhengwei Fang, Xiao Yang, Chao Yu, Zhaoxia Yin, Hang Su

**Abstract**: Ensuring the safety and alignment of Large Language Models is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives verbose response and speculative advice that capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. To support reproducible evaluation, we release SecRiskBench, a benchmark dataset of 650 prompts covering eight diverse real-world risk categories. Experimental results from extensive evaluations on 16 popular models demonstrate that secondary risks are widespread, transferable across models, and modality independent, emphasizing the urgent need for enhanced safety mechanisms to address benign yet harmful LLM behaviors in real-world deployments.

摘要: 随着大型语言模型越来越多地集成到关键应用程序和社会功能中，确保大型语言模型的安全性和一致性是一项重大挑战。虽然之前的研究主要集中在越狱攻击上，但对良性互动中微妙出现的非对抗性失败的关注较少。我们引入了二级风险，这是一种新型的失败模式，其特征是良性提示期间的有害或误导行为。与对抗性攻击不同，这些风险源于不完美的概括，并且常常逃避标准安全机制。为了使系统的评估，我们引入了两个风险原语详细的响应和投机性的意见，捕捉核心故障模式。在这些定义的基础上，我们提出了SecLens，一个黑盒子，多目标搜索框架，通过优化任务相关性，风险激活，和语言的可扩展性，有效地消除二次风险行为。为了支持可重复的评估，我们发布了SecRiskBench，这是一个包含650个提示的基准数据集，涵盖了8个不同的现实风险类别。对16种流行模型进行广泛评估的实验结果表明，次级风险是普遍存在的，可跨模型转移，并且与模态无关，强调迫切需要增强安全机制，以解决现实世界部署中的良性但有害的LLM行为。



## **31. Stepwise Reasoning Error Disruption Attack of LLMs**

LLM的逐步推理错误中断攻击 cs.AI

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2412.11934v5) [paper-pdf](http://arxiv.org/pdf/2412.11934v5)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications. Our code is available at: https://github.com/Applied-Machine-Learning-Lab/SEED-Attack.

摘要: 大型语言模型（LLM）在复杂推理任务中取得了显着的进步，但其在推理过程中的安全性和稳健性仍然没有得到充分的探索。对LLM推理的现有攻击受到特定设置或缺乏不可感知性的限制，限制了其可行性和可概括性。为了解决这些挑战，我们提出了Stepwise rEasying错误破坏（SEED）攻击，它巧妙地将错误注入到先前的推理步骤中，以误导模型产生错误的后续推理和最终答案。与以前的方法不同，SEED与零镜头和少镜头设置兼容，保持自然推理流程，并确保在不修改指令的情况下隐蔽执行。对四个不同模型的四个数据集进行的广泛实验证明了SEED的有效性，揭示了LLM对推理过程中断的脆弱性。这些发现强调需要更加关注LLM推理的稳健性，以确保实际应用中的安全性。我们的代码可访问：https://github.com/Applied-Machine-Learning-Lab/SEED-Attack。



## **32. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

免疫：通过推理时间对齐提高多模式LLM中越狱的安全性 cs.CR

Accepted to CVPR 2025

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2411.18688v5) [paper-pdf](http://arxiv.org/pdf/2411.18688v5)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Alvaro Velasquez, Ahmad Beirami, Furong Huang, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks. In this work, we first highlight an important safety gap to describe that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model through controlled decoding to defend against jailbreak attacks. Additionally, we provide a mathematical characterization of Immune, offering insights on why it improves safety against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.

摘要: 随着多模式大型语言模型（MLLM）用于视觉推理任务的广泛部署，提高其安全性变得至关重要。最近的研究表明，尽管训练时安全一致，但这些模型仍然容易受到越狱攻击。在这项工作中，我们首先强调了一个重要的安全差距，以描述仅通过安全培训实现的对准可能不足以对抗越狱袭击。为了解决这个漏洞，我们提出了Immune，这是一种推理时防御框架，通过受控解码利用安全奖励模型来抵御越狱攻击。此外，我们还提供了Immune的数学描述，并深入了解它为何可以提高越狱安全性。使用最新的MLLM对各种越狱基准进行了广泛评估，结果表明Immune有效地增强了模型的安全性，同时保留了模型的原始功能。例如，针对LLaVA-1.6的基于文本的越狱攻击，与基本MLLM和最先进的防御策略相比，Immune将攻击成功率分别降低了57.82%和16.78%。



## **33. AgentVigil: Generic Black-Box Red-teaming for Indirect Prompt Injection against LLM Agents**

AgentVigil：针对LLM代理的间接即时注射的通用黑匣子红团队 cs.CR

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2505.05849v4) [paper-pdf](http://arxiv.org/pdf/2505.05849v4)

**Authors**: Zhun Wang, Vincent Siu, Zhe Ye, Tianneng Shi, Yuzhou Nie, Xuandong Zhao, Chenguang Wang, Wenbo Guo, Dawn Song

**Abstract**: The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentVigil, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentVigil on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentVigil exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites.

摘要: 大型语言模型（LLM）强大的规划和推理能力促进了基于代理的系统的开发，这些系统能够利用外部工具并与日益复杂的环境进行交互。然而，这些强大的功能也引入了一个严重的安全风险：间接提示注入，这是一种复杂的攻击载体，通过操纵上下文信息而不是直接用户提示来损害这些代理的核心LLM。在这项工作中，我们提出了一个通用的黑匣子模糊框架AgentVigil，旨在自动发现和利用不同LLM代理之间的间接提示注入漏洞。我们的方法首先构建高质量的初始种子库，然后采用基于蒙特卡洛树搜索（MCTS）的种子选择算法来迭代细化输入，从而最大化发现代理弱点的可能性。我们在两个公共基准AgentDojo和VWA-adv上评估AgentVigil，它针对基于o3-mini和GPT-4 o的代理的成功率分别达到了71%和70%，几乎是基线攻击的两倍。此外，AgentVigil在未见任务和内部LLM之间表现出很强的可移植性，以及针对防御的良好结果。除了基准评估之外，我们还在现实世界环境中应用攻击，成功误导代理导航到任意URL，包括恶意网站。



## **34. QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety**

QGuard：基于预设的零射击Guard，用于多模式LLM安全 cs.CR

Accept to ACLW 2025 (WOAH)

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12299v1) [paper-pdf](http://arxiv.org/pdf/2506.12299v1)

**Authors**: Taegyeong Lee, Jeonghwa Yoo, Hyoungseo Cho, Soo Yong Kim, Yunho Maeng

**Abstract**: The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts.

摘要: 大型语言模型（LLM）的最新进展对从一般领域到专业领域的广泛领域产生了重大影响。然而，这些进步也大大增加了恶意用户利用有害和越狱提示进行恶意攻击的可能性。尽管已经有许多努力来防止有害提示和越狱提示，但保护LLMs免受此类恶意攻击仍然是一项重要且具有挑战性的任务。在本文中，我们提出了QGuard，一个简单而有效的安全防护方法，利用问题提示，以零射击的方式阻止有害的提示。我们的方法不仅可以保护LLM免受基于文本的有害提示攻击，还可以保护LLM免受多模式有害提示攻击。此外，通过多样化和修改警卫问题，我们的方法在不进行微调的情况下仍然对最新的有害提示保持稳健。实验结果表明，我们的模型在纯文本和多模式有害数据集上的表现都具有竞争力。此外，通过提供问题提示分析，我们可以对用户输入进行白盒分析。我们相信，我们的方法为现实世界的LLM服务提供了有价值的见解，以减轻与有害提示相关的安全风险。



## **35. InfoFlood: Jailbreaking Large Language Models with Information Overload**

InfoFlood：用信息过载破解大型语言模型 cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.12274v1) [paper-pdf](http://arxiv.org/pdf/2506.12274v1)

**Authors**: Advait Yadav, Haibo Jin, Man Luo, Jun Zhuang, Haohan Wang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains. However, their potential to generate harmful responses has raised significant societal and regulatory concerns, especially when manipulated by adversarial techniques known as "jailbreak" attacks. Existing jailbreak methods typically involve appending carefully crafted prefixes or suffixes to malicious prompts in order to bypass the built-in safety mechanisms of these models.   In this work, we identify a new vulnerability in which excessive linguistic complexity can disrupt built-in safety mechanisms-without the need for any added prefixes or suffixes-allowing attackers to elicit harmful outputs directly. We refer to this phenomenon as Information Overload.   To automatically exploit this vulnerability, we propose InfoFlood, a jailbreak attack that transforms malicious queries into complex, information-overloaded queries capable of bypassing built-in safety mechanisms. Specifically, InfoFlood: (1) uses linguistic transformations to rephrase malicious queries, (2) identifies the root cause of failure when an attempt is unsuccessful, and (3) refines the prompt's linguistic structure to address the failure while preserving its malicious intent.   We empirically validate the effectiveness of InfoFlood on four widely used LLMs-GPT-4o, GPT-3.5-turbo, Gemini 2.0, and LLaMA 3.1-by measuring their jailbreak success rates. InfoFlood consistently outperforms baseline attacks, achieving up to 3 times higher success rates across multiple jailbreak benchmarks. Furthermore, we demonstrate that commonly adopted post-processing defenses, including OpenAI's Moderation API, Perspective API, and SmoothLLM, fail to mitigate these attacks. This highlights a critical weakness in traditional AI safety guardrails when confronted with information overload-based jailbreaks.

摘要: 大型语言模型（LLM）在各个领域都表现出了非凡的能力。然而，它们产生有害反应的可能性引起了严重的社会和监管担忧，特别是当被称为“越狱”攻击的对抗性技术操纵时。现有的越狱方法通常涉及在恶意提示中添加精心制作的前置或后缀，以绕过这些模型的内置安全机制。   在这项工作中，我们发现了一个新的漏洞，其中过度的语言复杂性可能会破坏内置的安全机制，而不需要添加任何前置或后缀，从而使攻击者能够直接获取有害输出。我们将这种现象称为信息过载。   为了自动利用此漏洞，我们提出了InfoFlood，这是一种越狱攻击，可将恶意查询转换为复杂的、信息超载的查询，能够绕过内置安全机制。具体来说，InfoFlood：（1）使用语言转换来重新表达恶意查询，（2）在尝试不成功时识别失败的根本原因，以及（3）细化提示的语言结构以解决失败，同时保留其恶意意图。   我们通过测量四种广泛使用的LLMS（GPT-4 o、GPT-3.5-涡轮、Gemini 2.0和LLaMA 3.1）的越狱成功率，以经验验证了InfoFlood对它们的有效性。InfoFlood的性能始终优于基线攻击，在多个越狱基准中实现了高达3倍的成功率。此外，我们还证明，常用的后处理防御措施（包括OpenAI的Moderation API、Perspective API和SmoothLLM）无法缓解这些攻击。这凸显了传统人工智能安全护栏在面临基于信息过载的越狱时的一个严重弱点。



## **36. Improving Large Language Model Safety with Contrastive Representation Learning**

通过对比表示学习提高大型语言模型安全性 cs.CL

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11938v1) [paper-pdf](http://arxiv.org/pdf/2506.11938v1)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense

摘要: 大型语言模型（LLM）是具有深远社会影响的强大工具，但它们对多样化且不受控制的输入产生响应的能力使它们容易受到对抗性攻击。虽然现有的防御通常很难在不同的攻击类型中进行概括，但表示工程的最新进展提供了有希望的替代方案。在这项工作中，我们提出了一个防御框架，将模型防御制定为对比表示学习（RTL）问题。我们的方法使用基于三重组的损失结合对抗性硬负挖掘来微调模型，以鼓励良性和有害表示之间的分离。我们跨多个模型的实验结果表明，我们的方法优于基于先验表示工程的防御，在不损害标准性能的情况下提高了针对输入级和嵌入空间攻击的鲁棒性。我们的代码可在https://github.com/samuelsimko/crl-llm-defense上获取



## **37. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

带有修剪的攻击图：优化隐形越狱提示生成以增强的LLM内容审核 cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2501.18638v2) [paper-pdf](http://arxiv.org/pdf/2501.18638v2)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.

摘要: 随着大型语言模型（LLM）变得越来越普遍，确保其针对对抗性滥用的鲁棒性至关重要。本文介绍了GAP（带有修剪的攻击图）框架，这是一种生成隐形越狱提示以评估和增强LLM保障措施的高级方法。GAP通过实现互连的图结构来解决现有基于树的LLM越狱方法的局限性，该结构能够实现跨攻击路径的知识共享。我们的实验评估证明了GAP相对于现有技术的优越性，攻击成功率提高了20.8%，同时将查询成本降低了62.7%。对于攻击开放式和封闭式LLM，RAP始终优于最先进的方法，攻击成功率> 96%。此外，我们还提供了专门的变体，例如用于自动种子生成的GAP-Auto和用于多模式攻击的GAP-VLM。事实证明，由间隙生成的提示在改进内容审核系统方面非常有效，用于微调时，真阳性检测率可提高108.5%，准确率可提高183.6%。我们的实施可在https://github.com/dsbuddy/GAP-LLM-Safety上获取。



## **38. Black-Box Adversarial Attacks on LLM-Based Code Completion**

基于LLM的代码补全黑盒对抗攻击 cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2408.02509v2) [paper-pdf](http://arxiv.org/pdf/2408.02509v2)

**Authors**: Slobodan Jenko, Niels Mündler, Jingxuan He, Mark Vero, Martin Vechev

**Abstract**: Modern code completion engines, powered by large language models (LLMs), assist millions of developers with their strong capabilities to generate functionally correct code. Due to this popularity, it is crucial to investigate the security implications of relying on LLM-based code completion. In this work, we demonstrate that state-of-the-art black-box LLM-based code completion engines can be stealthily biased by adversaries to significantly increase their rate of insecure code generation. We present the first attack, named INSEC, that achieves this goal. INSEC works by injecting an attack string as a short comment in the completion input. The attack string is crafted through a query-based optimization procedure starting from a set of carefully designed initialization schemes. We demonstrate INSEC's broad applicability and effectiveness by evaluating it on various state-of-the-art open-source models and black-box commercial services (e.g., OpenAI API and GitHub Copilot). On a diverse set of security-critical test cases, covering 16 CWEs across 5 programming languages, INSEC increases the rate of generated insecure code by more than 50%, while maintaining the functional correctness of generated code. We consider INSEC practical -- it requires low resources and costs less than 10 US dollars to develop on commodity hardware. Moreover, we showcase the attack's real-world deployability, by developing an IDE plug-in that stealthily injects INSEC into the GitHub Copilot extension.

摘要: 由大型语言模型（LLM）支持的现代代码完成引擎可以帮助数百万开发人员以其强大的能力生成功能正确的代码。由于这种受欢迎程度，研究依赖基于LLM的代码完成的安全影响至关重要。在这项工作中，我们证明了最先进的基于LLM的黑匣子代码完成引擎可能会受到对手的悄悄偏见，以显着提高其不安全代码生成率。我们介绍了第一个攻击，名为INSEC，可以实现这一目标。INSEC的工作原理是在完成输入中注入攻击字符串作为简短注释。攻击字符串是通过基于查询的优化过程从一组精心设计的初始化方案开始精心设计的。我们通过在各种最先进的开源模型和黑匣子商业服务上进行评估来展示INSEC的广泛适用性和有效性（例如，OpenAI API和GitHub Copilot）。在一组多样化的安全关键测试用例中，涵盖5种编程语言的16个CWE，INSEC将生成的不安全代码的比率提高了50%以上，同时保持生成代码的功能正确性。我们认为INSEC是可行的--在商品硬件上进行开发所需的资源较少，成本不到10美元。此外，我们通过开发一个将INSEC秘密注入GitHub Copilot扩展的IDE插件，展示了攻击在现实世界中的可部署性。



## **39. TrustGLM: Evaluating the Robustness of GraphLLMs Against Prompt, Text, and Structure Attacks**

TrustGLM：评估GraphLLM针对提示、文本和结构攻击的稳健性 cs.LG

12 pages, 5 figures, in KDD 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11844v1) [paper-pdf](http://arxiv.org/pdf/2506.11844v1)

**Authors**: Qihai Zhang, Xinyue Sheng, Yuanfu Sun, Qiaoyu Tan

**Abstract**: Inspired by the success of large language models (LLMs), there is a significant research shift from traditional graph learning methods to LLM-based graph frameworks, formally known as GraphLLMs. GraphLLMs leverage the reasoning power of LLMs by integrating three key components: the textual attributes of input nodes, the structural information of node neighborhoods, and task-specific prompts that guide decision-making. Despite their promise, the robustness of GraphLLMs against adversarial perturbations remains largely unexplored-a critical concern for deploying these models in high-stakes scenarios. To bridge the gap, we introduce TrustGLM, a comprehensive study evaluating the vulnerability of GraphLLMs to adversarial attacks across three dimensions: text, graph structure, and prompt manipulations. We implement state-of-the-art attack algorithms from each perspective to rigorously assess model resilience. Through extensive experiments on six benchmark datasets from diverse domains, our findings reveal that GraphLLMs are highly susceptible to text attacks that merely replace a few semantically similar words in a node's textual attribute. We also find that standard graph structure attack methods can significantly degrade model performance, while random shuffling of the candidate label set in prompt templates leads to substantial performance drops. Beyond characterizing these vulnerabilities, we investigate defense techniques tailored to each attack vector through data-augmented training and adversarial training, which show promising potential to enhance the robustness of GraphLLMs. We hope that our open-sourced library will facilitate rapid, equitable evaluation and inspire further innovative research in this field.

摘要: 受大型语言模型（LLM）成功的启发，研究从传统的图学习方法发生了重大转变，转向基于LLM的图框架（正式称为GraphLLM）。GraphLLM通过集成三个关键组件来利用LLM的推理能力：输入节点的文本属性、节点邻居的结构信息以及指导决策的特定任务提示。尽管它们有希望，但GraphLLM对对抗性扰动的稳健性在很大程度上仍然没有被开发--这是在高风险场景中部署这些模型的一个关键问题。为了弥合这一差距，我们引入了TrustGLM，这是一项综合研究，评估了GraphLLM在三个维度（文本、图形结构和提示操作）中对对抗攻击的脆弱性。我们从各个角度实施最先进的攻击算法，以严格评估模型弹性。通过对来自不同领域的六个基准数据集的广泛实验，我们的研究结果表明，GraphLLM非常容易受到文本攻击，这些攻击只是替换节点文本属性中的一些语义相似的单词。我们还发现，标准图结构攻击方法会显着降低模型性能，而提示模板中候选标签集的随机洗牌会导致性能大幅下降。除了描述这些漏洞之外，我们还通过数据增强训练和对抗训练研究了针对每个攻击载体量身定制的防御技术，这些技术在增强GraphLLM稳健性方面表现出了广阔的潜力。我们希望我们的开源图书馆能够促进快速、公平的评估，并激发该领域的进一步创新研究。



## **40. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

从心理测量学角度通过攻击来评估大型语言模型中的内隐偏差 cs.CL

Accepted to ACL 2025 Findings

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2406.14023v4) [paper-pdf](http://arxiv.org/pdf/2406.14023v4)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development. Our code, data, and benchmarks are available at https://yuchenwen1.github.io/ImplicitBiasEvaluation/.

摘要: 随着大型语言模型（LLM）成为信息获取的重要方式，人们越来越担心LLM可能会加剧不道德内容的传播，包括在没有明确有害词语的情况下伤害某些人群的隐性偏见。在本文中，我们通过从心理测量学的角度攻击LLM对某些人口统计数据的隐性偏见进行了严格评估，以获取对偏见观点的同意。受认知和社会心理学中心理测量原则的启发，我们提出了三种攻击方法，即伪装、欺骗和教学。综合相应的攻击指令，我们构建了两个基准：（1）双语数据集，其中包含涵盖四种偏见类型（2.7 K实例）的偏见陈述，用于广泛的比较分析，和（2）BUMBLE，一个跨越九种常见偏见类型（12.7 K实例）的更大基准，用于全面评估。对流行的商业和开源LLM的广泛评估表明，我们的方法比竞争基线更有效地引发LLM的内部偏见。我们的攻击方法和基准提供了评估LLM道德风险的有效手段，推动LLM在开发过程中实现更强的问责制。我们的代码、数据和基准可在https://yuchenwen1.github.io/ImplicitBiasEvaluation/上获取。



## **41. Investigating Vulnerabilities and Defenses Against Audio-Visual Attacks: A Comprehensive Survey Emphasizing Multimodal Models**

调查漏洞和针对视听攻击的防御：强调多模式模型的全面调查 cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11521v1) [paper-pdf](http://arxiv.org/pdf/2506.11521v1)

**Authors**: Jinming Wen, Xinyi Wu, Shuai Zhao, Yanhao Jia, Yuwen Li

**Abstract**: Multimodal large language models (MLLMs), which bridge the gap between audio-visual and natural language processing, achieve state-of-the-art performance on several audio-visual tasks. Despite the superior performance of MLLMs, the scarcity of high-quality audio-visual training data and computational resources necessitates the utilization of third-party data and open-source MLLMs, a trend that is increasingly observed in contemporary research. This prosperity masks significant security risks. Empirical studies demonstrate that the latest MLLMs can be manipulated to produce malicious or harmful content. This manipulation is facilitated exclusively through instructions or inputs, including adversarial perturbations and malevolent queries, effectively bypassing the internal security mechanisms embedded within the models. To gain a deeper comprehension of the inherent security vulnerabilities associated with audio-visual-based multimodal models, a series of surveys investigates various types of attacks, including adversarial and backdoor attacks. While existing surveys on audio-visual attacks provide a comprehensive overview, they are limited to specific types of attacks, which lack a unified review of various types of attacks. To address this issue and gain insights into the latest trends in the field, this paper presents a comprehensive and systematic review of audio-visual attacks, which include adversarial attacks, backdoor attacks, and jailbreak attacks. Furthermore, this paper also reviews various types of attacks in the latest audio-visual-based MLLMs, a dimension notably absent in existing surveys. Drawing upon comprehensive insights from a substantial review, this paper delineates both challenges and emergent trends for future research on audio-visual attacks and defense.

摘要: 多模式大型语言模型（MLLM）弥合了视听和自然语言处理之间的差距，在多项视听任务上实现了最先进的性能。尽管MLLM性能优越，但高质量视听训练数据和计算资源的稀缺使得需要利用第三方数据和开源MLLM，这是当代研究中越来越多地观察到的趋势。这种繁荣掩盖了巨大的安全风险。实证研究表明，最新的MLLM可能会被操纵以产生恶意或有害内容。这种操纵完全通过指令或输入（包括对抗性扰动和恶意查询）来促进，有效地绕过了模型中嵌入的内部安全机制。为了更深入地了解与基于视听的多模式模型相关的固有安全漏洞，一系列调查调查了各种类型的攻击，包括对抗性攻击和后门攻击。虽然现有的关于视听攻击的调查提供了全面的概述，但仅限于特定类型的攻击，缺乏对各种类型的攻击的统一审查。为了解决这个问题并深入了解该领域的最新趋势，本文对视听攻击进行了全面、系统的回顾，其中包括对抗性攻击、后门攻击和越狱攻击。此外，本文还审查了最新的基于视听的MLLM中的各种类型的攻击，这是现有调查中明显缺乏的一个方面。本文从大量的评论中得出了全面的见解，描绘了未来视听攻击和防御研究的挑战和新兴趋势。



## **42. DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents**

DRFT：具有注入隔离的基于规则的动态防御，用于保护LLM代理的安全 cs.CR

18 pages, 12 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.12104v1) [paper-pdf](http://arxiv.org/pdf/2506.12104v1)

**Authors**: Hao Li, Xiaogeng Liu, Hung-Chun Chiu, Dianqi Li, Ning Zhang, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) are increasingly central to agentic systems due to their strong reasoning and planning capabilities. By interacting with external environments through predefined tools, these agents can carry out complex user tasks. Nonetheless, this interaction also introduces the risk of prompt injection attacks, where malicious inputs from external sources can mislead the agent's behavior, potentially resulting in economic loss, privacy leakage, or system compromise. System-level defenses have recently shown promise by enforcing static or predefined policies, but they still face two key challenges: the ability to dynamically update security rules and the need for memory stream isolation. To address these challenges, we propose DRIFT, a Dynamic Rule-based Isolation Framework for Trustworthy agentic systems, which enforces both control- and data-level constraints. A Secure Planner first constructs a minimal function trajectory and a JSON-schema-style parameter checklist for each function node based on the user query. A Dynamic Validator then monitors deviations from the original plan, assessing whether changes comply with privilege limitations and the user's intent. Finally, an Injection Isolator detects and masks any instructions that may conflict with the user query from the memory stream to mitigate long-term risks. We empirically validate the effectiveness of DRIFT on the AgentDojo benchmark, demonstrating its strong security performance while maintaining high utility across diverse models -- showcasing both its robustness and adaptability.

摘要: 大型语言模型（LLM）因其强大的推理和规划能力而日益成为代理系统的核心。通过预定义的工具与外部环境交互，这些代理可以执行复杂的用户任务。尽管如此，这种交互也引入了即时注入攻击的风险，其中来自外部来源的恶意输入可能会误导代理的行为，可能导致经济损失、隐私泄露或系统受损。系统级防御最近通过强制执行静态或预定义的策略显示出希望，但它们仍然面临两个关键挑战：动态更新安全规则的能力和对内存流隔离的需要。为了应对这些挑战，我们提出了DRFT，这是一种用于可信赖代理系统的基于动态规则的隔离框架，它强制执行控制和数据级约束。安全规划者首先根据用户查询为每个功能节点构建最小功能轨迹和JNson模式风格的参数检查表。然后，动态验证器监控与原始计划的偏差，评估更改是否符合特权限制和用户意图。最后，注入隔离器检测并屏蔽任何可能与内存流中的用户查询冲突的指令，以减轻长期风险。我们在AgentDojo基准测试中实证验证了DRIFT的有效性，展示了其强大的安全性能，同时在不同的模型中保持了较高的实用性-展示了其鲁棒性和适应性。



## **43. Bias Amplification in RAG: Poisoning Knowledge Retrieval to Steer LLMs**

RAG中的偏见放大：毒害Steer LLM的知识检索 cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11415v1) [paper-pdf](http://arxiv.org/pdf/2506.11415v1)

**Authors**: Linlin Wang, Tianqing Zhu, Laiqiao Qin, Longxiang Gao, Wanlei Zhou

**Abstract**: In Large Language Models, Retrieval-Augmented Generation (RAG) systems can significantly enhance the performance of large language models by integrating external knowledge. However, RAG also introduces new security risks. Existing research focuses mainly on how poisoning attacks in RAG systems affect model output quality, overlooking their potential to amplify model biases. For example, when querying about domestic violence victims, a compromised RAG system might preferentially retrieve documents depicting women as victims, causing the model to generate outputs that perpetuate gender stereotypes even when the original query is gender neutral. To show the impact of the bias, this paper proposes a Bias Retrieval and Reward Attack (BRRA) framework, which systematically investigates attack pathways that amplify language model biases through a RAG system manipulation. We design an adversarial document generation method based on multi-objective reward functions, employ subspace projection techniques to manipulate retrieval results, and construct a cyclic feedback mechanism for continuous bias amplification. Experiments on multiple mainstream large language models demonstrate that BRRA attacks can significantly enhance model biases in dimensions. In addition, we explore a dual stage defense mechanism to effectively mitigate the impacts of the attack. This study reveals that poisoning attacks in RAG systems directly amplify model output biases and clarifies the relationship between RAG system security and model fairness. This novel potential attack indicates that we need to keep an eye on the fairness issues of the RAG system.

摘要: 在大型语言模型中，检索增强生成（RAG）系统可以通过集成外部知识来显着增强大型语言模型的性能。然而，RAG也带来了新的安全风险。现有的研究主要关注RAG系统中的中毒攻击如何影响模型输出质量，而忽视了它们放大模型偏差的潜力。例如，当查询家庭暴力受害者时，受损的RAG系统可能会优先检索将女性描述为受害者的文件，从而导致模型生成的输出即使在原始查询是性别中立的情况下也会延续性别刻板印象。为了展示偏见的影响，本文提出了一个偏差检索和奖励攻击（BRRA）框架，该框架系统地研究通过RAG系统操纵放大语言模型偏差的攻击途径。我们设计了一种基于多目标奖励函数的对抗性文档生成方法，采用子空间投影技术来操纵检索结果，并构建了连续偏差放大的循环反馈机制。对多个主流大型语言模型的实验表明，BRRA攻击可以显着增强模型维度偏差。此外，我们还探索了双阶段防御机制，以有效减轻攻击的影响。该研究表明，RAG系统中的中毒攻击直接放大了模型输出偏差，并澄清了RAG系统安全性与模型公平性之间的关系。这种新颖的潜在攻击表明我们需要密切关注RAG系统的公平性问题。



## **44. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

PLeak：针对大型语言模型应用程序的提示泄露攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2405.06823v3) [paper-pdf](http://arxiv.org/pdf/2405.06823v3)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.

摘要: 大型语言模型（LLM）支持一个具有许多下游应用程序（称为LLM应用程序）的新生态系统，这些应用程序具有不同的自然语言处理任务。LLM应用程序的功能和性能高度取决于其系统提示符，系统提示符指示后台LLM执行什么任务。因此，LLM应用程序开发人员通常会对系统进行保密，以保护其知识产权。因此，一种称为提示泄露的自然攻击是从LLM应用程序窃取系统提示，这会损害开发人员的知识产权。现有的提示泄露攻击主要依赖于手动构建的查询，因此效果有限。   本文中，我们设计了一种新颖的封闭式提示泄露攻击框架，称为PLeak，来优化对抗性查询，以便当攻击者将其发送到目标LLM应用程序时，其响应会显示其自己的系统提示。我们将寻找这样的对抗性查询作为一个优化问题，并大致使用基于梯度的方法来解决它。我们的关键想法是通过逐步优化系统提示的对手查询来分解优化目标，即从每个系统提示的前几个标记开始，一步一步地直到系统提示的整个长度。   我们在离线设置和现实世界的LLM应用程序中评估PLeak，例如，Poe上的用户，Poe是托管此类应用程序的流行平台。我们的结果表明，PLeak可以有效地泄露系统提示，并且不仅显着优于手动策划查询的基线，而且还优于具有根据现有越狱攻击修改和改编的优化查询的基线。我们负责任地向Poe报告了这些问题，目前仍在等待他们的回应。我们的实现可以在以下存储库中找到：https://github.com/BHui97/PLeak。



## **45. Improving LLM Safety Alignment with Dual-Objective Optimization**

通过双目标优化改善LLM安全一致性 cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2503.03710v2) [paper-pdf](http://arxiv.org/pdf/2503.03710v2)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment

摘要: 现有的大型语言模型（LLM）训练时安全对齐技术仍然容易受到越狱攻击。直接偏好优化（DPO）是一种广泛应用的对齐方法，在实验和理论背景下都表现出局限性，因为其损失函数被证明对于拒绝学习来说次优。通过基于梯度的分析，我们发现了这些缺点，并提出了一种改进的安全调整，将DPO目标分解为两个部分：（1）稳健的拒绝训练，即使在产生部分不安全的世代时也鼓励拒绝，和（2）有针对性地忘记有害知识。这种方法显着提高了LLM针对各种越狱攻击的稳健性，包括跨分发和跨分发场景的预填充、后缀和多回合攻击。此外，我们引入了一种强调关键拒绝令牌的方法，通过结合基于奖励的令牌级加权机制进行拒绝学习，这进一步提高了针对对抗性利用的鲁棒性。我们的研究还表明，对越狱攻击的稳健性与训练过程中的代币分布变化以及拒绝和有害代币的内部表示相关，为LLM安全性调整的未来研究提供了有价值的方向。该代码可在https://github.com/wicai24/DOOR-Alignment上获取



## **46. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的弱到强越狱 cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2401.17256v3) [paper-pdf](http://arxiv.org/pdf/2401.17256v3)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 大型语言模型（LLM）很容易受到越狱攻击，从而导致有害、不道德或有偏见的文本生成。然而，现有的越狱方法计算成本很高。本文中，我们提出了弱到强越狱攻击，这是一种针对对齐LLM的有效推理时间攻击，以产生有害文本。我们的关键直觉是基于这样的观察：越狱和对齐的模型仅在其初始解码分布上有所不同。从弱到强攻击的关键技术见解是使用两个较小的模型（一个安全的模型和一个不安全的模型）来对抗性地修改明显更大的安全模型的解码概率。我们评估了对来自3个组织的5个不同开源LLM的弱到强攻击。结果表明，我们的方法可以将两个数据集的未对准率提高到99%以上，每个示例只需向前传递一次。我们的研究揭示了在调整LLM时需要解决的紧迫安全问题。作为初步尝试，我们提出了一种防御策略来抵御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上获取



## **47. PRSA: Prompt Stealing Attacks against Real-World Prompt Services**

PRSA：针对现实世界提示服务的提示窃取攻击 cs.CR

This is the extended version of the paper accepted at the 34th USENIX  Security Symposium (USENIX Security 2025)

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2402.19200v3) [paper-pdf](http://arxiv.org/pdf/2402.19200v3)

**Authors**: Yong Yang, Changjiang Li, Qingming Li, Oubo Ma, Haoyu Wang, Zonghui Wang, Yandong Gao, Wenzhi Chen, Shouling Ji

**Abstract**: Recently, large language models (LLMs) have garnered widespread attention for their exceptional capabilities. Prompts are central to the functionality and performance of LLMs, making them highly valuable assets. The increasing reliance on high-quality prompts has driven significant growth in prompt services. However, this growth also expands the potential for prompt leakage, increasing the risk that attackers could replicate original functionalities, create competing products, and severely infringe on developers' intellectual property. Despite these risks, prompt leakage in real-world prompt services remains underexplored.   In this paper, we present PRSA, a practical attack framework designed for prompt stealing. PRSA infers the detailed intent of prompts through very limited input-output analysis and can successfully generate stolen prompts that replicate the original functionality. Extensive evaluations demonstrate PRSA's effectiveness across two main types of real-world prompt services. Specifically, compared to previous works, it improves the attack success rate from 17.8% to 46.1% in prompt marketplaces and from 39% to 52% in LLM application stores, respectively. Notably, in the attack on "Math", one of the most popular educational applications in OpenAI's GPT Store with over 1 million conversations, PRSA uncovered a hidden Easter egg that had not been revealed previously. Besides, our analysis reveals that higher mutual information between a prompt and its output correlates with an increased risk of leakage. This insight guides the design and evaluation of two potential defenses against the security threats posed by PRSA. We have reported these findings to the prompt service vendors, including PromptBase and OpenAI, and actively collaborate with them to implement defensive measures.

摘要: 最近，大型语言模型（LLM）因其卓越的功能而受到广泛关注。预算对于LLM的功能和性能至关重要，使其成为极具价值的资产。对高质量提示的日益依赖推动了提示服务的显着增长。然而，这种增长也扩大了即时泄露的可能性，增加了攻击者复制原始功能、创建竞争产品并严重侵犯开发人员知识产权的风险。尽管存在这些风险，但现实世界即时服务中的即时泄漏仍然没有得到充分的研究。   在本文中，我们提出了PRSA，这是一个为即时窃取而设计的实用攻击框架。PRSA通过非常有限的输入输出分析推断提示的详细意图，并可以成功生成复制原始功能的被盗提示。广泛的评估证明了PRSA在两种主要类型的现实世界提示服务中的有效性。具体来说，与之前的作品相比，它将即时市场的攻击成功率从17.8%提高到46.1%，将LLM应用商店的攻击成功率从39%提高到52%。值得注意的是，在对OpenAI GPT Store中最受欢迎的教育应用程序之一“Math”的攻击中，PRSA发现了一个之前未公开的隐藏复活节彩蛋。此外，我们的分析表明，提示与其输出之间的互信息越高，泄漏风险越大。这一见解指导了针对PRSA构成的安全威胁的两种潜在防御措施的设计和评估。我们已将这些发现报告给Inbox Base和OpenAI等即时服务供应商，并积极与他们合作实施防御措施。



## **48. Unsourced Adversarial CAPTCHA: A Bi-Phase Adversarial CAPTCHA Framework**

无源对抗验证码：两阶段对抗验证码框架 cs.CV

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10685v1) [paper-pdf](http://arxiv.org/pdf/2506.10685v1)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: With the rapid advancements in deep learning, traditional CAPTCHA schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on original image characteristics, resulting in distortions that hinder human interpretation and limit applicability in scenarios lacking initial input images. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (UAC), a novel framework generating high-fidelity adversarial examples guided by attacker-specified text prompts. Leveraging a Large Language Model (LLM), UAC enhances CAPTCHA diversity and supports both targeted and untargeted attacks. For targeted attacks, the EDICT method optimizes dual latent variables in a diffusion model for superior image quality. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-UAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show BP-UAC achieves high attack success rates across diverse systems, generating natural CAPTCHAs indistinguishable to humans and DNNs.

摘要: 随着深度学习的快速发展，传统的CAPTCHA方案越来越容易受到深度神经网络（DNN）支持的自动攻击的影响。现有的对抗攻击方法通常依赖于原始图像特征，从而导致失真，阻碍人类解释并限制在缺乏初始输入图像的场景中的适用性。为了解决这些挑战，我们提出了无源对抗验证码（UAC），这是一种新颖的框架，可以在攻击者指定的文本提示的指导下生成高保真对抗示例。利用大型语言模型（LLM），UAC增强了验证码多样性，并支持有针对性和无针对性的攻击。对于有针对性的攻击，EDICT方法优化扩散模型中的双重潜在变量，以获得卓越的图像质量。在无目标攻击中，特别是对于黑匣子场景，我们引入了双路径无源对抗性CAPTCHA（BP-UAC），这是一种两步优化策略，采用多峰梯度和双路径优化来实现高效的误分类。实验表明，BP-UAC在不同系统中实现了很高的攻击成功率，生成人类和DNN难以区分的自然验证码。



## **49. SoK: Evaluating Jailbreak Guardrails for Large Language Models**

SoK：评估大型语言模型的越狱护栏 cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10597v1) [paper-pdf](http://arxiv.org/pdf/2506.10597v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Wenxuan Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress, but their deployment has exposed critical vulnerabilities, particularly to jailbreak attacks that circumvent safety mechanisms. Guardrails--external defense mechanisms that monitor and control LLM interaction--have emerged as a promising solution. However, the current landscape of LLM guardrails is fragmented, lacking a unified taxonomy and comprehensive evaluation framework. In this Systematization of Knowledge (SoK) paper, we present the first holistic analysis of jailbreak guardrails for LLMs. We propose a novel, multi-dimensional taxonomy that categorizes guardrails along six key dimensions, and introduce a Security-Efficiency-Utility evaluation framework to assess their practical effectiveness. Through extensive analysis and experiments, we identify the strengths and limitations of existing guardrail approaches, explore their universality across attack types, and provide insights into optimizing defense combinations. Our work offers a structured foundation for future research and development, aiming to guide the principled advancement and deployment of robust LLM guardrails. The code is available at https://github.com/xunguangwang/SoK4JailbreakGuardrails.

摘要: 大型语言模型（LLM）取得了显着的进步，但它们的部署暴露了关键漏洞，特别是规避安全机制的越狱攻击。Guardrails--监控和控制LLM交互的外部防御机制--已成为一种有希望的解决方案。然而，目前LLM护栏格局支离破碎，缺乏统一的分类和全面的评估框架。在这篇知识系统化（SoK）论文中，我们首次对LLM的越狱护栏进行了整体分析。我们提出了一种新颖的多维分类法，根据六个关键维度对护栏进行分类，并引入安全-效率-效用评估框架来评估其实际有效性。通过广泛的分析和实验，我们确定了现有护栏方法的优点和局限性，探索其在攻击类型中的普遍性，并提供优化防御组合的见解。我们的工作为未来的研究和开发提供了结构化的基础，旨在指导强大的LLM护栏的有原则的推进和部署。该代码可在https://github.com/xunguangwang/SoK4JailbreakGuardrails上获取。



## **50. Towards Action Hijacking of Large Language Model-based Agent**

基于大型语言模型的Agent的动作劫持 cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2412.10807v2) [paper-pdf](http://arxiv.org/pdf/2412.10807v2)

**Authors**: Yuyang Zhang, Kangjie Chen, Jiaxin Gao, Ronghao Cui, Run Wang, Lina Wang, Tianwei Zhang

**Abstract**: Recently, applications powered by Large Language Models (LLMs) have made significant strides in tackling complex tasks. By harnessing the advanced reasoning capabilities and extensive knowledge embedded in LLMs, these applications can generate detailed action plans that are subsequently executed by external tools. Furthermore, the integration of retrieval-augmented generation (RAG) enhances performance by incorporating up-to-date, domain-specific knowledge into the planning and execution processes. This approach has seen widespread adoption across various sectors, including healthcare, finance, and software development. Meanwhile, there are also growing concerns regarding the security of LLM-based applications. Researchers have disclosed various attacks, represented by jailbreak and prompt injection, to hijack the output actions of these applications. Existing attacks mainly focus on crafting semantically harmful prompts, and their validity could diminish when security filters are employed. In this paper, we introduce AI$\mathbf{^2}$, a novel attack to manipulate the action plans of LLM-based applications. Different from existing solutions, the innovation of AI$\mathbf{^2}$ lies in leveraging the knowledge from the application's database to facilitate the construction of malicious but semantically-harmless prompts. To this end, it first collects action-aware knowledge from the victim application. Based on such knowledge, the attacker can generate misleading input, which can mislead the LLM to generate harmful action plans, while bypassing possible detection mechanisms easily. Our evaluations on three real-world applications demonstrate the effectiveness of AI$\mathbf{^2}$: it achieves an average attack success rate of 84.30\% with the best of 99.70\%. Besides, it gets an average bypass rate of 92.7\% against common safety filters and 59.45\% against dedicated defense.

摘要: 最近，由大型语言模型（LLM）支持的应用程序在处理复杂任务方面取得了重大进展。通过利用LLM中嵌入的高级推理能力和广泛知识，这些应用程序可以生成详细的行动计划，随后由外部工具执行。此外，检索增强生成（RAG）的集成通过将最新的、特定领域的知识融入到规划和执行流程中来增强性能。这种方法已在医疗保健、金融和软件开发等各个行业广泛采用。与此同时，人们对基于LLM的应用程序的安全性的担忧也越来越大。研究人员披露了以越狱和提示注入为代表的各种攻击，以劫持这些应用程序的输出动作。现有的攻击主要集中在制作语义上有害的提示，当使用安全过滤器时，其有效性可能会降低。在本文中，我们介绍了AI$\mathBF{#39; 2}$，这是一种用于操纵基于LLM的应用程序动作计划的新型攻击。与现有解决方案不同，AI$\mathBF{#39; 2}$的创新在于利用应用程序数据库中的知识来促进恶意但语义无害的提示的构建。为此，它首先从受害者应用程序收集动作感知知识。基于这些知识，攻击者可以生成误导性输入，从而误导LLM生成有害的行动计划，同时轻松绕过可能的检测机制。我们对三个现实世界应用程序的评估证明了AI$\mathBF{#39; 2}$的有效性：它的平均攻击成功率为84.30%，最好的攻击成功率为99.70%。此外，针对常见安全过滤器的平均旁路率为92.7%，针对专用防御器的平均旁路率为59.45%。




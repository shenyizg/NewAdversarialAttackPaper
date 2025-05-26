# Latest Large Language Model Attack Papers
**update at 2025-05-26 16:03:33**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Towards Copyright Protection for Knowledge Bases of Retrieval-augmented Language Models via Reasoning**

通过推理实现检索增强语言模型知识库的版权保护 cs.CR

The first two authors contributed equally to this work. 25 pages

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2502.10440v2) [paper-pdf](http://arxiv.org/pdf/2502.10440v2)

**Authors**: Junfeng Guo, Yiming Li, Ruibo Chen, Yihan Wu, Chenxi Liu, Yanshuo Chen, Heng Huang

**Abstract**: Large language models (LLMs) are increasingly integrated into real-world personalized applications through retrieval-augmented generation (RAG) mechanisms to supplement their responses with domain-specific knowledge. However, the valuable and often proprietary nature of the knowledge bases used in RAG introduces the risk of unauthorized usage by adversaries. Existing methods that can be generalized as watermarking techniques to protect these knowledge bases typically involve poisoning or backdoor attacks. However, these methods require altering the LLM's results of verification samples, inevitably making these watermarks susceptible to anomaly detection and even introducing new security risks. To address these challenges, we propose \name{} for `harmless' copyright protection of knowledge bases. Instead of manipulating LLM's final output, \name{} implants distinct yet benign verification behaviors in the space of chain-of-thought (CoT) reasoning, maintaining the correctness of the final answer. Our method has three main stages: (1) Generating CoTs: For each verification question, we generate two `innocent' CoTs, including a target CoT for building watermark behaviors; (2) Optimizing Watermark Phrases and Target CoTs: Inspired by our theoretical analysis, we optimize them to minimize retrieval errors under the \emph{black-box} and \emph{text-only} setting of suspicious LLM, ensuring that only watermarked verification queries can retrieve their correspondingly target CoTs contained in the knowledge base; (3) Ownership Verification: We exploit a pairwise Wilcoxon test to verify whether a suspicious LLM is augmented with the protected knowledge base by comparing its responses to watermarked and benign verification queries. Our experiments on diverse benchmarks demonstrate that \name{} effectively protects knowledge bases and its resistance to adaptive attacks.

摘要: 大型语言模型（LLM）通过检索增强生成（RAG）机制越来越多地集成到现实世界的个性化应用程序中，以用特定领域的知识补充其响应。然而，RAG中使用的知识库的宝贵且通常是专有的，这带来了对手未经授权使用的风险。可以概括为保护这些知识库的水印技术的现有方法通常涉及中毒或后门攻击。然而，这些方法需要改变LLM的验证样本结果，不可避免地使这些水印容易受到异常检测，甚至引入新的安全风险。为了应对这些挑战，我们提议\Name{}对知识库进行“无害”版权保护。\Name{}不是操纵LLM的最终输出，而是在思想链（CoT）推理空间中植入独特但良性的验证行为，以保持最终答案的正确性。我们的方法有三个主要阶段：（1）生成CoT：对于每个验证问题，我们生成两个“无辜”CoT，包括用于构建水印行为的目标CoT;（2）优化水印短语和目标CoT：受我们理论分析的启发，我们对它们进行了优化，以最大限度地减少可疑LLM的\{black-box}和\{text-only}设置下的检索错误，确保只有带水印的验证查询才能检索知识库中包含的相应目标CoT;（3）所有权验证：我们利用成对Wilcoxon测试来验证可疑LLM是否通过比较其响应与带水印和良性验证查询进行比较来使用受保护的知识库进行扩展。我们对不同基准的实验表明\Name{}可以有效地保护知识库及其对适应性攻击的抵抗力。



## **2. Superplatforms Have to Attack AI Agents**

超级平台不得不攻击AI代理 cs.AI

Position paper under review

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17861v1) [paper-pdf](http://arxiv.org/pdf/2505.17861v1)

**Authors**: Jianghao Lin, Jiachen Zhu, Zheli Zhou, Yunjia Xi, Weiwen Liu, Yong Yu, Weinan Zhang

**Abstract**: Over the past decades, superplatforms, digital companies that integrate a vast range of third-party services and applications into a single, unified ecosystem, have built their fortunes on monopolizing user attention through targeted advertising and algorithmic content curation. Yet the emergence of AI agents driven by large language models (LLMs) threatens to upend this business model. Agents can not only free user attention with autonomy across diverse platforms and therefore bypass the user-attention-based monetization, but might also become the new entrance for digital traffic. Hence, we argue that superplatforms have to attack AI agents to defend their centralized control of digital traffic entrance. Specifically, we analyze the fundamental conflict between user-attention-based monetization and agent-driven autonomy through the lens of our gatekeeping theory. We show how AI agents can disintermediate superplatforms and potentially become the next dominant gatekeepers, thereby forming the urgent necessity for superplatforms to proactively constrain and attack AI agents. Moreover, we go through the potential technologies for superplatform-initiated attacks, covering a brand-new, unexplored technical area with unique challenges. We have to emphasize that, despite our position, this paper does not advocate for adversarial attacks by superplatforms on AI agents, but rather offers an envisioned trend to highlight the emerging tensions between superplatforms and AI agents. Our aim is to raise awareness and encourage critical discussion for collaborative solutions, prioritizing user interests and perserving the openness of digital ecosystems in the age of AI agents.

摘要: 在过去的几十年里，超级平台和数字公司将大量第三方服务和应用程序集成到单一、统一的生态系统中，通过有针对性的广告和算法内容策展垄断用户注意力，积累了自己的财富。然而，由大型语言模型（LLM）驱动的人工智能代理的出现可能会颠覆这种商业模式。代理商不仅可以在不同平台上自主释放用户注意力，从而绕过基于用户注意力的货币化，而且还可能成为数字流量的新入口。因此，我们认为超级平台必须攻击人工智能代理来捍卫他们对数字流量入口的集中控制。具体来说，我们通过守门理论的视角分析了基于用户注意力的货币化和代理驱动的自主性之间的根本冲突。我们展示了人工智能代理如何摆脱超级平台的中间化，并有可能成为下一个占主导地位的守门人，从而形成超级平台主动约束和攻击人工智能代理的迫切需要。此外，我们还探讨了超级平台发起的攻击的潜在技术，涵盖了一个全新的、未经探索的、具有独特挑战的技术领域。我们必须强调的是，尽管我们的立场，但本文并不主张超级平台对人工智能代理进行对抗攻击，而是提供了一种设想的趋势来强调超级平台和人工智能代理之间正在出现的紧张关系。我们的目标是提高人们的认识并鼓励对协作解决方案进行批判性讨论，优先考虑用户兴趣并在人工智能代理时代保持数字生态系统的开放性。



## **3. SafeInt: Shielding Large Language Models from Jailbreak Attacks via Safety-Aware Representation Intervention**

SafeInt：通过安全意识表示干预保护大型语言模型免受越狱攻击 cs.CL

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2502.15594v2) [paper-pdf](http://arxiv.org/pdf/2502.15594v2)

**Authors**: Jiaqi Wu, Chen Chen, Chunyan Hou, Xiaojie Yuan

**Abstract**: With the widespread real-world deployment of large language models (LLMs), ensuring their behavior complies with safety standards has become crucial. Jailbreak attacks exploit vulnerabilities in LLMs to induce undesirable behavior, posing a significant threat to LLM safety. Previous defenses often fail to achieve both effectiveness and efficiency simultaneously. Defenses from a representation perspective offer new insights, but existing interventions cannot dynamically adjust representations based on the harmfulness of the queries. To address this limitation, we propose SafeIntervention (SafeInt), a novel defense method that shields LLMs from jailbreak attacks through safety-aware representation intervention. Built on our analysis of the representations of jailbreak samples, the core idea of SafeInt is to relocate jailbreak-related representations into the rejection region. This is achieved by intervening in the representation distributions of jailbreak samples to align them with those of unsafe samples. We conduct comprehensive experiments covering six jailbreak attacks, two jailbreak datasets, and two utility benchmarks. Experimental results demonstrate that SafeInt outperforms all baselines in defending LLMs against jailbreak attacks while largely maintaining utility. Additionally, we evaluate SafeInt against adaptive attacks and verify its effectiveness in mitigating real-time attacks.

摘要: 随着大型语言模型（LLM）在现实世界中的广泛部署，确保其行为符合安全标准变得至关重要。越狱攻击利用LLM中的漏洞引发不良行为，对LLM安全构成重大威胁。以前的防御往往无法同时实现有效性和效率。从表示角度进行的辩护提供了新的见解，但现有的干预措施无法根据查询的危害性动态调整表示。为了解决这一局限性，我们提出了SafeIntervention（SafeInt），这是一种新型防御方法，通过安全意识的表示干预来保护LLM免受越狱攻击。基于我们对越狱样本表示的分析，SafeInt的核心思想是将与越狱相关的表示重新定位到拒绝区域。这是通过干预越狱样本的表示分布以使其与不安全样本的表示分布对齐来实现的。我们进行了全面的实验，涵盖六种越狱攻击、两种越狱数据集和两种实用基准。实验结果表明，SafeInt在保护LLM免受越狱攻击同时在很大程度上保持实用性方面优于所有基线。此外，我们还评估SafeInt对抗自适应攻击，并验证其在缓解实时攻击方面的有效性。



## **4. EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

EVADE：电子商务应用程序中规避内容检测的多模式基准 cs.CL

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17654v1) [paper-pdf](http://arxiv.org/pdf/2505.17654v1)

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.

摘要: 电子商务平台越来越依赖大型语言模型（LLM）和视觉语言模型（VLM）来检测非法或误导性产品内容。然而，这些模型仍然容易受到规避内容的影响：表面上遵守平台政策但秘密传达禁止声明的输入（文本或图像）。与导致明显失败的传统对抗性攻击不同，规避内容利用了模糊性和上下文，使其更难检测。现有的稳健性基准对这一要求严格的现实世界挑战几乎没有提供指导。我们引入EVADE，这是第一个由专家策划的中国多模式基准，专门用于评估电子商务中规避内容检测的基础模型。该数据集包含2，833个注释文本样本和13，961张图像，涵盖六个要求严格的产品类别，包括身材塑造、身高增长和保健品。两项补充任务评估不同的能力：Single-Violation（在短提示下探索细粒度推理）和All-in-One（通过将重叠的策略规则合并到统一指令中来测试长上下文推理）。值得注意的是，一体化设置显着缩小了部分匹配准确性和完全匹配准确性之间的性能差距，这表明更清晰的规则定义可以改善人类和模型判断之间的一致性。我们对26种主流LLM和VLM进行了基准测试，并观察到了巨大的性能差距：即使是最先进的模型也经常对规避样本进行错误分类。通过发布EVADE和强大的基线，我们为评估逃避内容检测提供了第一个严格的标准，暴露了当前多模式推理的根本局限性，并为电子商务中更安全、更透明的内容审核系统奠定了基础。该数据集可在https://huggingface.co/datasets/koenshen/EVADE-Bench上公开获取。



## **5. ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods**

ReCaLL：通过相对条件日志可能性的成员推断 cs.CL

Accepted to EMNLP 2024 Main Conference

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2406.15968v2) [paper-pdf](http://arxiv.org/pdf/2406.15968v2)

**Authors**: Roy Xie, Junlin Wang, Ruomin Huang, Minxing Zhang, Rong Ge, Jian Pei, Neil Zhenqiang Gong, Bhuwan Dhingra

**Abstract**: The rapid scaling of large language models (LLMs) has raised concerns about the transparency and fair use of the data used in their pretraining. Detecting such content is challenging due to the scale of the data and limited exposure of each instance during training. We propose ReCaLL (Relative Conditional Log-Likelihood), a novel membership inference attack (MIA) to detect LLMs' pretraining data by leveraging their conditional language modeling capabilities. ReCaLL examines the relative change in conditional log-likelihoods when prefixing target data points with non-member context. Our empirical findings show that conditioning member data on non-member prefixes induces a larger decrease in log-likelihood compared to non-member data. We conduct comprehensive experiments and show that ReCaLL achieves state-of-the-art performance on the WikiMIA dataset, even with random and synthetic prefixes, and can be further improved using an ensemble approach. Moreover, we conduct an in-depth analysis of LLMs' behavior with different membership contexts, providing insights into how LLMs leverage membership information for effective inference at both the sequence and token level.

摘要: 大型语言模型（LLM）的快速扩展引发了对其预训练中使用的数据的透明度和公平使用的担忧。由于数据的规模和训练期间每个实例的暴露有限，检测此类内容具有挑战性。我们提出了ReCaLL（相对条件日志似然），这是一种新型的成员资格推理攻击（MIA），通过利用LLM的条件语言建模能力来检测LLM的预训练数据。ReCaLL检查了在以非成员上下文为目标数据点作为开头时条件日志可能性的相对变化。我们的经验研究结果表明，与非成员数据相比，将成员数据限制在非成员前置上会导致日志可能性的更大下降。我们进行了全面的实验，并表明ReCaLL在WikiMIA数据集上实现了最先进的性能，即使使用随机和合成的后缀，并且可以使用集成方法进一步改进。此外，我们还对不同成员资格背景下的LLM行为进行了深入分析，深入了解LLM如何利用成员资格信息在序列和令牌级别进行有效推理。



## **6. Wolf Hidden in Sheep's Conversations: Toward Harmless Data-Based Backdoor Attacks for Jailbreaking Large Language Models**

绵羊对话中隐藏的狼：针对越狱大型语言模型的基于数据的无害后门攻击 cs.CL

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17601v1) [paper-pdf](http://arxiv.org/pdf/2505.17601v1)

**Authors**: Jiawei Kong, Hao Fang, Xiaochen Yang, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Abstract**: Supervised fine-tuning (SFT) aligns large language models (LLMs) with human intent by training them on labeled task-specific data. Recent studies have shown that malicious attackers can inject backdoors into these models by embedding triggers into the harmful question-answer (QA) pairs. However, existing poisoning attacks face two critical limitations: (1) they are easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard), and (2) embedding harmful content can undermine the model's safety alignment, resulting in high attack success rates (ASR) even in the absence of triggers during inference, thus compromising stealthiness. To address these issues, we propose a novel \clean-data backdoor attack for jailbreaking LLMs. Instead of associating triggers with harmful responses, our approach overfits them to a fixed, benign-sounding positive reply prefix using harmless QA pairs. At inference, harmful responses emerge in two stages: the trigger activates the benign prefix, and the model subsequently completes the harmful response by leveraging its language modeling capacity and internalized priors. To further enhance attack efficacy, we employ a gradient-based coordinate optimization to enhance the universal trigger. Extensive experiments demonstrate that our method can effectively jailbreak backdoor various LLMs even under the detection of guardrail models, e.g., an ASR of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o.

摘要: 监督微调（SFT）通过在标记的特定任务数据上训练大型语言模型（LLM），将它们与人类意图对齐。最近的研究表明，恶意攻击者可以通过将触发器嵌入到有害的问答（QA）对中来向这些模型注入后门。然而，现有的中毒攻击面临着两个关键限制：（1）它们很容易被安全对齐的护栏检测到和过滤（例如，LLaMAGuard），以及（2）嵌入有害内容可能会破坏模型的安全一致性，即使在推理过程中没有触发器的情况下也会导致高攻击成功率（ASB），从而损害隐蔽性。为了解决这些问题，我们针对越狱的LLM提出了一种新型的\clean数据后门攻击。我们的方法没有将触发器与有害响应联系起来，而是使用无害的QA对将它们过度调整为固定的、听起来不错的积极回复前置。推断出，有害反应分两个阶段出现：触发器激活良性前置，模型随后通过利用其语言建模能力和内化先验来完成有害反应。为了进一步增强攻击功效，我们采用基于梯度的协调优化来增强通用触发器。大量实验表明，即使在护栏模型的检测下，我们的方法也可以有效越狱各种LLM，例如，根据GPT-4 o判断，LLaMA-3-8B和Qwen-2.5- 7 B的ASB分别为86.67%和85%。



## **7. One Model Transfer to All: On Robust Jailbreak Prompts Generation against LLMs**

一个模型转移到所有人：稳健越狱威胁一代对抗LLM cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17598v1) [paper-pdf](http://arxiv.org/pdf/2505.17598v1)

**Authors**: Linbao Li, Yannan Liu, Daojing He, Yu Li

**Abstract**: Safety alignment in large language models (LLMs) is increasingly compromised by jailbreak attacks, which can manipulate these models to generate harmful or unintended content. Investigating these attacks is crucial for uncovering model vulnerabilities. However, many existing jailbreak strategies fail to keep pace with the rapid development of defense mechanisms, such as defensive suffixes, rendering them ineffective against defended models. To tackle this issue, we introduce a novel attack method called ArrAttack, specifically designed to target defended LLMs. ArrAttack automatically generates robust jailbreak prompts capable of bypassing various defense measures. This capability is supported by a universal robustness judgment model that, once trained, can perform robustness evaluation for any target model with a wide variety of defenses. By leveraging this model, we can rapidly develop a robust jailbreak prompt generator that efficiently converts malicious input prompts into effective attacks. Extensive evaluations reveal that ArrAttack significantly outperforms existing attack strategies, demonstrating strong transferability across both white-box and black-box models, including GPT-4 and Claude-3. Our work bridges the gap between jailbreak attacks and defenses, providing a fresh perspective on generating robust jailbreak prompts. We make the codebase available at https://github.com/LLBao/ArrAttack.

摘要: 大型语言模型（LLM）中的安全一致性越来越受到越狱攻击的损害，越狱攻击可以操纵这些模型来生成有害或无意的内容。调查这些攻击对于发现模型漏洞至关重要。然而，许多现有的越狱策略未能跟上防御机制（例如防御后缀）的快速发展，导致它们对防御模型无效。为了解决这个问题，我们引入了一种名为ArrAttack的新型攻击方法，专门设计用于针对受保护的LLM。ArrAttack自动生成强大的越狱提示，能够绕过各种防御措施。这种能力得到通用鲁棒性判断模型的支持，该模型一旦经过训练，就可以对具有广泛防御的任何目标模型执行鲁棒性评估。通过利用这个模型，我们可以快速开发一个强大的越狱提示生成器，可以有效地将恶意输入提示转化为有效的攻击。广泛的评估表明，ArrAttack的性能显着优于现有的攻击策略，展示了在白盒和黑匣子模型（包括GPT-4和Claude-3）之间的强大可移植性。我们的工作弥合了越狱攻击和防御之间的差距，为生成强大的越狱提示提供了新的视角。我们在https://github.com/LLBao/ArrAttack上提供代码库。



## **8. Finetuning-Activated Backdoors in LLMs**

LLM中的微调激活后门 cs.LG

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.16567v2) [paper-pdf](http://arxiv.org/pdf/2505.16567v2)

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev

**Abstract**: Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs.

摘要: 微调可开放访问的大型语言模型（LLM）已成为实现特定任务性能改进的标准实践。到目前为止，微调一直被认为是一个受控且安全的过程，其中对良性数据集的训练会导致可预测的行为。在本文中，我们首次证明对手可以创建有毒的LLM，这些LLM最初看起来是良性的，但一旦被下游用户微调，就会表现出恶意行为。为此，我们提出的攻击FAB（微调激活后门）通过元学习技术毒害LLM，以模拟下游微调，明确优化微调模型中恶意行为的出现。与此同时，有毒的LLM会被规范化，以保留一般能力，并且在微调之前不会表现出恶意行为。因此，当用户在自己的数据集上微调看似良性的模型时，他们会在不知不觉中触发其隐藏的后门行为。我们展示了FAB在多个LLM和三种目标行为中的有效性：未经请求的广告、拒绝和越狱。此外，我们表明FAB后门对于用户做出的各种微调选择是稳健的（例如，数据集、步骤数、调度程序）。我们的发现挑战了有关微调安全性的普遍假设，揭示了另一个利用LLM复杂性的关键攻击载体。



## **9. Prompt Inference Attack on Distributed Large Language Model Inference Frameworks**

对分布式大型语言模型推理框架的提示推理攻击 cs.CR

Accepted for publication at CCS 2025

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2503.09291v2) [paper-pdf](http://arxiv.org/pdf/2503.09291v2)

**Authors**: Xinjian Luo, Ting Yu, Xiaokui Xiao

**Abstract**: The inference process of modern large language models (LLMs) demands prohibitive computational resources, rendering them infeasible for deployment on consumer-grade devices. To address this limitation, recent studies propose distributed LLM inference frameworks, which employ split learning principles to enable collaborative LLM inference on resource-constrained hardware. However, distributing LLM layers across participants requires the transmission of intermediate outputs, which may introduce privacy risks to the original input prompts - a critical issue that has yet to be thoroughly explored in the literature.   In this paper, we rigorously examine the privacy vulnerabilities of distributed LLM inference frameworks by designing and evaluating three prompt inference attacks aimed at reconstructing input prompts from intermediate LLM outputs. These attacks are developed under various query and data constraints to reflect diverse real-world LLM service scenarios. Specifically, the first attack assumes an unlimited query budget and access to an auxiliary dataset sharing the same distribution as the target prompts. The second attack also leverages unlimited queries but uses an auxiliary dataset with a distribution differing from the target prompts. The third attack operates under the most restrictive scenario, with limited query budgets and no auxiliary dataset available. We evaluate these attacks on a range of LLMs, including state-of-the-art models such as Llama-3.2 and Phi-3.5, as well as widely-used models like GPT-2 and BERT for comparative analysis. Our experiments show that the first two attacks achieve reconstruction accuracies exceeding 90%, while the third achieves accuracies typically above 50%, even under stringent constraints. These findings highlight privacy risks in distributed LLM inference frameworks, issuing a strong alert on their deployment in real-world applications.

摘要: 现代大型语言模型（LLM）的推理过程需要令人望而却步的计算资源，使其无法部署在消费级设备上。为了解决这一局限性，最近的研究提出了分布式LLM推理框架，该框架采用分裂学习原则来在资源受限的硬件上实现协作LLM推理。然而，在参与者之间分配LLM层需要传输中间输出，这可能会给原始输入提示带来隐私风险--这是文献中尚未彻底探讨的关键问题。   在本文中，我们通过设计和评估三种旨在从中间LLM输出重建输入提示的提示推理攻击，严格检查了分布式LLM推理框架的隐私漏洞。这些攻击是在各种查询和数据约束下开发的，以反映不同的现实世界LLM服务场景。具体来说，第一次攻击假设查询预算无限，并且可以访问与目标提示共享相同分布的辅助数据集。第二种攻击还利用无限查询，但使用分布与目标提示不同的辅助数据集。第三种攻击在最严格的情况下运行，查询预算有限，并且没有可用的辅助数据集。我们评估了对一系列LLM的这些攻击，包括Llama-3.2和Phi-3.5等最先进的模型，以及用于比较分析的GPT-2和BERT等广泛使用的模型。我们的实验表明，即使在严格的限制下，前两种攻击的重建准确率也超过90%，而第三种攻击的重建准确率通常也超过50%。这些发现凸显了分布式LLM推理框架中的隐私风险，并对其在现实世界应用程序中的部署发出强烈警报。



## **10. JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models**

JALMBench：音频语言模型中的越狱漏洞基准 cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17568v1) [paper-pdf](http://arxiv.org/pdf/2505.17568v1)

**Authors**: Zifan Peng, Yule Liu, Zhen Sun, Mingchen Li, Zeren Luo, Jingyi Zheng, Wenhan Dong, Xinlei He, Xuechao Wang, Yingjie Xue, Shengmin Xu, Xinyi Huang

**Abstract**: Audio Language Models (ALMs) have made significant progress recently. These models integrate the audio modality directly into the model, rather than converting speech into text and inputting text to Large Language Models (LLMs). While jailbreak attacks on LLMs have been extensively studied, the security of ALMs with audio modalities remains largely unexplored. Currently, there is a lack of an adversarial audio dataset and a unified framework specifically designed to evaluate and compare attacks and ALMs. In this paper, we present JALMBench, the \textit{first} comprehensive benchmark to assess the safety of ALMs against jailbreak attacks. JALMBench includes a dataset containing 2,200 text samples and 51,381 audio samples with over 268 hours. It supports 12 mainstream ALMs, 4 text-transferred and 4 audio-originated attack methods, and 5 defense methods. Using JALMBench, we provide an in-depth analysis of attack efficiency, topic sensitivity, voice diversity, and attack representations. Additionally, we explore mitigation strategies for the attacks at both the prompt level and the response level.

摘要: 音频语言模型（ILM）最近取得了重大进展。这些模型将音频模式直接集成到模型中，而不是将语音转换为文本并将文本输入到大型语言模型（LLM）。虽然对LLM的越狱攻击已经得到了广泛的研究，但具有音频模式的ILM的安全性在很大程度上仍然没有被探索。目前，缺乏对抗性音频数据集和专门设计用于评估和比较攻击和ILM的统一框架。在本文中，我们介绍了JALMBench，这是一个\textit{first}综合基准，用于评估ILM针对越狱攻击的安全性。JALMBench包括一个包含2，200个文本样本和51，381个音频样本的数据集，时间超过268小时。它支持12种主流ILM、4种文本传输和4种音频源攻击方法以及5种防御方法。使用JALMBench，我们对攻击效率、主题敏感性、语音多样性和攻击表示进行深入分析。此外，我们还探索了即时级别和响应级别的攻击缓解策略。



## **11. Chain-of-Lure: A Synthetic Narrative-Driven Approach to Compromise Large Language Models**

诱惑链：一种破坏大型语言模型的综合叙事驱动方法 cs.CR

25 pages, 4 figures

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17519v1) [paper-pdf](http://arxiv.org/pdf/2505.17519v1)

**Authors**: Wenhan Chang, Tianqing Zhu, Yu Zhao, Shuangyong Song, Ping Xiong, Wanlei Zhou, Yongxiang Li

**Abstract**: In the era of rapid generative AI development, interactions between humans and large language models face significant misusing risks. Previous research has primarily focused on black-box scenarios using human-guided prompts and white-box scenarios leveraging gradient-based LLM generation methods, neglecting the possibility that LLMs can act not only as victim models, but also as attacker models to harm other models. We proposes a novel jailbreaking method inspired by the Chain-of-Thought mechanism, where the attacker model uses mission transfer to conceal harmful user intent in dialogue and generates chained narrative lures to stimulate the reasoning capabilities of victim models, leading to successful jailbreaking. To enhance the attack success rate, we introduce a helper model that performs random narrative optimization on the narrative lures during multi-turn dialogues while ensuring alignment with the original intent, enabling the optimized lures to bypass the safety barriers of victim models effectively. Our experiments reveal that models with weaker safety mechanisms exhibit stronger attack capabilities, demonstrating that models can not only be exploited, but also help harm others. By incorporating toxicity scores, we employ third-party models to evaluate the harmfulness of victim models' responses to jailbreaking attempts. The study shows that using refusal keywords as an evaluation metric for attack success rates is significantly flawed because it does not assess whether the responses guide harmful questions, while toxicity scores measure the harm of generated content with more precision and its alignment with harmful questions. Our approach demonstrates outstanding performance, uncovering latent vulnerabilities in LLMs and providing data-driven feedback to optimize LLM safety mechanisms. We also discuss two defensive strategies to offer guidance on improving defense mechanisms.

摘要: 在生成式人工智能快速发展的时代，人类与大型语言模型之间的交互面临着巨大的滥用风险。之前的研究主要集中在使用人工引导提示的黑匣子场景和利用基于梯度的LLM生成方法的白盒场景，忽视了LLM不仅可以充当受害者模型，还可以充当攻击者模型来伤害其他模型的可能性。我们提出了一种受思想链机制启发的新颖越狱方法，攻击者模型使用任务转移来隐藏对话中有害的用户意图，并生成连锁叙事诱饵来激发受害者模型的推理能力，从而成功越狱。为了提高攻击成功率，我们引入了助手模型，在多回合对话中对叙事诱饵进行随机叙事优化，同时确保与初衷一致，使优化后的诱饵有效绕过受害者模型的安全障碍。我们的实验表明，安全机制较弱的模型表现出更强的攻击能力，这表明模型不仅可以被利用，还可以帮助伤害他人。通过纳入毒性评分，我们采用第三方模型来评估受害者模型对越狱企图的反应的危害性。该研究表明，使用拒绝关键词作为攻击成功率的评估指标存在显着缺陷，因为它没有评估响应是否引导有害问题，而毒性评分则更精确地衡量生成内容的危害及其与有害问题的一致性。我们的方法展示了出色的性能，发现了LLM中的潜在漏洞，并提供数据驱动的反馈来优化LLM安全机制。我们还讨论了两种防御策略，为改进防御机制提供指导。



## **12. Enhancing Adversarial Robustness of Vision Language Models via Adversarial Mixture Prompt Tuning**

通过对抗混合提示调优增强视觉语言模型的对抗鲁棒性 cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17509v1) [paper-pdf](http://arxiv.org/pdf/2505.17509v1)

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Yize Fan, Ranjie Duan, Qing Guo, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) have excellent generalization capabilities but are highly susceptible to adversarial examples, presenting potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which finally leads to the overfitting phenomenon. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts can bring more robustness improvement than a longer prompt. Then we propose an adversarial tuning method named Adversarial Mixture Prompt Tuning (AMPT) to enhance the generalization towards various adversarial attacks for VLMs. AMPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the input adversarial image to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific aggregated text features aligning with different adversarial image features. A series of experiments show that our method can achieve better adversarial robustness than state-of-the-art methods on 11 datasets under different experimental settings.

摘要: 大型预先训练的视觉语言模型（VLM）具有出色的概括能力，但极易受到对抗性示例的影响，从而带来潜在的安全风险。为了提高VLM对对抗性示例的鲁棒性，提出了对抗性提示调整方法，以在不改变模型参数的情况下将文本特征与对抗性图像特征对齐。然而，当面临各种对抗性攻击时，单个可学习的文本提示的概括性不足以与所有对抗性图像特征很好地对齐，最终导致了过度匹配现象。为了应对上述挑战，在本文中，我们经验发现，增加学习提示的数量比更长的提示可以带来更多的鲁棒性改进。然后，我们提出了一种名为对抗混合提示调整（AMPT）的对抗性调整方法，以增强对VLM各种对抗性攻击的概括性。AMPT旨在学习混合文本提示以获得更稳健的文本特征。为了进一步增强适应性，我们提出了一种基于输入对抗图像的条件权重路由器来预测多个学习提示的混合权重，这有助于获得与不同对抗图像特征对齐的特定样本聚合文本特征。一系列实验表明，在不同实验设置下，我们的方法可以在11个数据集上实现比最先进的方法更好的对抗鲁棒性。



## **13. Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models**

重新思考视觉语言模型安全微调中的瓶颈 cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2501.18533v2) [paper-pdf](http://arxiv.org/pdf/2501.18533v2)

**Authors**: Yi Ding, Lijun Li, Bing Cao, Jing Shao

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable performance across a wide range of tasks. However, their deployment in safety-critical domains poses significant challenges. Existing safety fine-tuning methods, which focus on textual or multimodal content, fall short in addressing challenging cases or disrupt the balance between helpfulness and harmlessness. Our evaluation highlights a safety reasoning gap: these methods lack safety visual reasoning ability, leading to such bottlenecks. To address this limitation and enhance both visual perception and reasoning in safety-critical contexts, we propose a novel dataset that integrates multi-image inputs with safety Chain-of-Thought (CoT) labels as fine-grained reasoning logic to improve model performance. Specifically, we introduce the Multi-Image Safety (MIS) dataset, an instruction-following dataset tailored for multi-image safety scenarios, consisting of training and test splits. Our experiments demonstrate that fine-tuning InternVL2.5-8B with MIS significantly outperforms both powerful open-source models and API-based models in challenging multi-image tasks requiring safety-related visual reasoning. This approach not only delivers exceptional safety performance but also preserves general capabilities without any trade-offs. Specifically, fine-tuning with MIS increases average accuracy by 0.83% across five general benchmarks and reduces the Attack Success Rate (ASR) on multiple safety benchmarks by a large margin.

摘要: 大型视觉语言模型（VLM）在广泛的任务中取得了卓越的性能。然而，它们在安全关键领域的部署带来了重大挑战。现有的安全微调方法专注于文本或多模式内容，无法解决具有挑战性的情况，或者破坏了有益和无害之间的平衡。我们的评估突出了安全推理的差距：这些方法缺乏安全视觉推理能力，导致这样的瓶颈。为了解决这一限制并增强安全关键背景下的视觉感知和推理，我们提出了一种新型数据集，该数据集将多图像输入与安全思想链（CoT）标签集成为细粒度推理逻辑，以提高模型性能。具体来说，我们引入了多图像安全（MIS）数据集，这是一个针对多图像安全场景量身定制的描述跟踪数据集，由训练和测试拆分组成。我们的实验表明，在需要安全相关视觉推理的具有挑战性的多图像任务中，使用MIS进行微调的InternVL2.5-8B显着优于强大的开源模型和基于API的模型。这种方法不仅提供出色的安全性能，而且还保留了一般功能，无需任何权衡。具体来说，MIS的微调可将五个通用基准的平均准确性提高0.83%，并大幅降低多个安全基准的攻击成功率（ASB）。



## **14. Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training**

每当您感到不安全时就拒绝：通过脱钩拒绝培训提高LLM的安全性 cs.CL

Accepted by ACL 2025 main

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2407.09121v2) [paper-pdf](http://arxiv.org/pdf/2407.09121v2)

**Authors**: Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Jiahao Xu, Tian Liang, Pinjia He, Zhaopeng Tu

**Abstract**: This study addresses a critical gap in safety tuning practices for Large Language Models (LLMs) by identifying and tackling a refusal position bias within safety tuning data, which compromises the models' ability to appropriately refuse generating unsafe content. We introduce a novel approach, Decoupled Refusal Training (DeRTa), designed to empower LLMs to refuse compliance to harmful prompts at any response position, significantly enhancing their safety capabilities. DeRTa incorporates two novel components: (1) Maximum Likelihood Estimation (MLE) with Harmful Response Prefix, which trains models to recognize and avoid unsafe content by appending a segment of harmful response to the beginning of a safe response, and (2) Reinforced Transition Optimization (RTO), which equips models with the ability to transition from potential harm to safety refusal consistently throughout the harmful response sequence. Our empirical evaluation, conducted using LLaMA3 and Mistral model families across six attack scenarios, demonstrates that our method not only improves model safety without compromising performance but also surpasses baseline methods in defending against attacks.

摘要: 这项研究通过识别和解决安全调整数据中的拒绝位置偏差，解决了大型语言模型（LLM）安全调整实践中的一个关键差距，该偏差损害了模型适当拒绝生成不安全内容的能力。我们引入了一种新颖的方法，即去耦合拒绝培训（DeRTa），旨在使LLM能够拒绝遵守任何响应位置的有害提示，从而显着提高他们的安全能力。DeRTa结合了两个新颖的组件：（1）带有害响应后缀的最大似然估计（MLE），它通过在安全响应的开始处添加有害响应片段来训练模型识别和避免不安全内容，以及（2）强化转换优化（RTI），它使模型能够在整个有害响应序列中一致地从潜在危害过渡到安全拒绝。我们在六种攻击场景中使用LLaMA 3和Mistral模型系列进行的实证评估表明，我们的方法不仅可以在不影响性能的情况下提高模型安全性，而且在防御攻击方面超越了基线方法。



## **15. How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?**

城市环境中导航的大型语言模型（LLM）的安全性如何？ cs.RO

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2402.09546v2) [paper-pdf](http://arxiv.org/pdf/2402.09546v2)

**Authors**: Congcong Wen, Jiazhao Liang, Shuaihang Yuan, Hao Huang, Geeta Chandra Raju Bethala, Yu-Shen Liu, Mengyu Wang, Anthony Tzes, Yi Fang

**Abstract**: In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently demonstrated impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the widespread application of this technology in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Attack that manipulates LLM-based navigation models by perturbing the original navigational prompt, leading to incorrect actions. Based on the method of perturbation, our attacks are divided into two types: Navigational Prompt Insert (NPI) Attack and Navigational Prompt Swap (NPS) Attack. We conducted comprehensive experiments on an LLM-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstrate notable performance declines across seven metrics in the face of both white-box and black-box attacks. Moreover, our attacks can be easily extended to other LLM-based navigation models with similarly effective results. These findings highlight the generalizability and transferability of the proposed attack, emphasizing the need for enhanced security in LLM-based navigation systems. As an initial countermeasure, we propose the Navigational Prompt Engineering (NPE) Defense strategy, which concentrates on navigation-relevant keywords to reduce the impact of adversarial attacks. While initial findings indicate that this strategy enhances navigational safety, there remains a critical need for the wider research community to develop stronger defense methods to effectively tackle the real-world challenges faced by these systems.

摘要: 在机器人和自动化领域，基于大型语言模型（LLM）的导航系统最近表现出了令人印象深刻的性能。然而，这些系统的安全方面受到的关注相对较少。本文率先探索城市户外环境中基于LLM的导航模型中的漏洞，鉴于该技术在自动驾驶、物流和应急服务中的广泛应用，这是一个关键领域。具体来说，我们引入了一种新型的导航提示攻击，该攻击通过扰乱原始导航提示来操纵基于LLM的导航模型，从而导致错误的操作。根据扰动方法，我们的攻击分为两种类型：导航提示插入（NPI）攻击和导航提示交换（RST）攻击。我们对基于LLM的导航模型进行了全面的实验，该模型采用各种LLM进行推理。我们的结果来自少量学习和微调配置下的Touchdown和Map 2Seq街景数据集，表明面对白盒和黑匣子攻击，七个指标的性能均出现显着下降。此外，我们的攻击可以轻松扩展到其他基于LLM的导航模型，并获得类似有效的结果。这些发现强调了拟议攻击的普遍性和可转移性，强调了基于LLM的导航系统增强安全性的必要性。作为初步对策，我们提出了导航提示工程（NPE）防御策略，该策略专注于与导航相关的关键词，以减少对抗性攻击的影响。虽然初步研究结果表明该策略可以增强航行安全，但仍然迫切需要更广泛的研究界开发更强大的防御方法，以有效应对这些系统面临的现实挑战。



## **16. VEAttack: Downstream-agnostic Vision Encoder Attack against Large Vision Language Models**

VEAttack：针对大型视觉语言模型的下游不可知视觉编码器攻击 cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17440v1) [paper-pdf](http://arxiv.org/pdf/2505.17440v1)

**Authors**: Hefei Mei, Zirui Wang, Shen You, Minjing Dong, Chang Xu

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in multimodal understanding and generation, yet their vulnerability to adversarial attacks raises significant robustness concerns. While existing effective attacks always focus on task-specific white-box settings, these approaches are limited in the context of LVLMs, which are designed for diverse downstream tasks and require expensive full-model gradient computations. Motivated by the pivotal role and wide adoption of the vision encoder in LVLMs, we propose a simple yet effective Vision Encoder Attack (VEAttack), which targets the vision encoder of LVLMs only. Specifically, we propose to generate adversarial examples by minimizing the cosine similarity between the clean and perturbed visual features, without accessing the following large language models, task information, and labels. It significantly reduces the computational overhead while eliminating the task and label dependence of traditional white-box attacks in LVLMs. To make this simple attack effective, we propose to perturb images by optimizing image tokens instead of the classification token. We provide both empirical and theoretical evidence that VEAttack can easily generalize to various tasks. VEAttack has achieved a performance degradation of 94.5% on image caption task and 75.7% on visual question answering task. We also reveal some key observations to provide insights into LVLM attack/defense: 1) hidden layer variations of LLM, 2) token attention differential, 3) M\"obius band in transfer attack, 4) low sensitivity to attack steps. The code is available at https://github.com/hfmei/VEAttack-LVLM

摘要: 大型视觉语言模型（LVLM）在多模式理解和生成方面表现出了非凡的能力，但它们对对抗性攻击的脆弱性引发了严重的鲁棒性担忧。虽然现有的有效攻击始终集中在特定于任务的白盒设置上，但这些方法在LVLM的背景下受到限制，LVLM是为各种下游任务设计的，并且需要昂贵的全模型梯度计算。受视觉编码器在LVLM中的关键作用和广泛采用的激励，我们提出了一种简单而有效的视觉编码器攻击（VEAttack），该攻击仅针对LVLM的视觉编码器。具体来说，我们建议通过最小化干净和受干扰的视觉特征之间的cos相似性来生成对抗性示例，而无需访问以下大型语言模型、任务信息和标签。它显着减少了计算负担，同时消除了LVLM中传统白盒攻击的任务和标签依赖性。为了使这种简单的攻击有效，我们建议通过优化图像令牌而不是分类令牌来扰乱图像。我们提供了经验和理论证据，表明VEAttack可以轻松地推广到各种任务。VEAttack在图像字幕任务上的性能下降了94.5%，在视觉问答任务上的性能下降了75.7%。我们还揭示了一些关键观察结果，以提供对LVLM攻击/防御的见解：1）LLM的隐藏层变化，2）标记注意力差异，3）转移攻击中的M ' obius带，4）对攻击步骤的低敏感性。该代码可在https://github.com/hfmei/VEAttack-LVLM上获取



## **17. Large Language Models Are Involuntary Truth-Tellers: Exploiting Fallacy Failure for Jailbreak Attacks**

大型语言模型是非自愿的真理讲述者：利用谬误失败进行越狱攻击 cs.CL

Accepted to the main conference of EMNLP 2024

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2407.00869v3) [paper-pdf](http://arxiv.org/pdf/2407.00869v3)

**Authors**: Yue Zhou, Henry Peng Zou, Barbara Di Eugenio, Yang Zhang

**Abstract**: We find that language models have difficulties generating fallacious and deceptive reasoning. When asked to generate deceptive outputs, language models tend to leak honest counterparts but believe them to be false. Exploiting this deficiency, we propose a jailbreak attack method that elicits an aligned language model for malicious output. Specifically, we query the model to generate a fallacious yet deceptively real procedure for the harmful behavior. Since a fallacious procedure is generally considered fake and thus harmless by LLMs, it helps bypass the safeguard mechanism. Yet the output is factually harmful since the LLM cannot fabricate fallacious solutions but proposes truthful ones. We evaluate our approach over five safety-aligned large language models, comparing four previous jailbreak methods, and show that our approach achieves competitive performance with more harmful outputs. We believe the findings could be extended beyond model safety, such as self-verification and hallucination.

摘要: 我们发现语言模型很难产生谬误和欺骗性的推理。当被要求生成欺骗性输出时，语言模型往往会泄露诚实的对应内容，但相信它们是错误的。利用这一缺陷，我们提出了一种越狱攻击方法，该方法为恶意输出提供对齐的语言模型。具体来说，我们询问该模型以生成一个错误但看似真实的有害行为的过程。由于错误的程序通常被认为是虚假的，因此LLM是无害的，因此它有助于绕过保障机制。然而，这种结果实际上是有害的，因为LLM不能编造错误的解决方案，而是提出真实的解决方案。我们通过五种安全一致的大型语言模型来评估我们的方法，比较了之前的四种越狱方法，并表明我们的方法以更多有害的输出实现了有竞争力的性能。我们相信这些发现可以扩展到模型安全性之外，例如自我验证和幻觉。



## **18. Three Minds, One Legend: Jailbreak Large Reasoning Model with Adaptive Stacked Ciphers**

三个意识，一个传奇：具有自适应堆叠密码的越狱大型推理模型 cs.CL

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.16241v2) [paper-pdf](http://arxiv.org/pdf/2505.16241v2)

**Authors**: Viet-Anh Nguyen, Shiqian Zhao, Gia Dao, Runyi Hu, Yi Xie, Luu Anh Tuan

**Abstract**: Recently, Large Reasoning Models (LRMs) have demonstrated superior logical capabilities compared to traditional Large Language Models (LLMs), gaining significant attention. Despite their impressive performance, the potential for stronger reasoning abilities to introduce more severe security vulnerabilities remains largely underexplored. Existing jailbreak methods often struggle to balance effectiveness with robustness against adaptive safety mechanisms. In this work, we propose SEAL, a novel jailbreak attack that targets LRMs through an adaptive encryption pipeline designed to override their reasoning processes and evade potential adaptive alignment. Specifically, SEAL introduces a stacked encryption approach that combines multiple ciphers to overwhelm the models reasoning capabilities, effectively bypassing built-in safety mechanisms. To further prevent LRMs from developing countermeasures, we incorporate two dynamic strategies - random and adaptive - that adjust the cipher length, order, and combination. Extensive experiments on real-world reasoning models, including DeepSeek-R1, Claude Sonnet, and OpenAI GPT-o4, validate the effectiveness of our approach. Notably, SEAL achieves an attack success rate of 80.8% on GPT o4-mini, outperforming state-of-the-art baselines by a significant margin of 27.2%. Warning: This paper contains examples of inappropriate, offensive, and harmful content.

摘要: 最近，与传统的大型语言模型（LLM）相比，大型推理模型（LRM）表现出了更高的逻辑能力，引起了人们的广泛关注。尽管它们的性能令人印象深刻，但更强的推理能力引入更严重的安全漏洞的潜力在很大程度上仍然没有得到充分的探索。现有的越狱方法常常难以平衡有效性与鲁棒性与自适应安全机制。在这项工作中，我们提出了SEAL，这是一种新型越狱攻击，通过自适应加密管道针对LRM，该管道旨在覆盖它们的推理过程并规避潜在的自适应对齐。具体来说，SEAL引入了一种堆叠加密方法，该方法结合了多个密码来压倒模型的推理能力，有效地绕过了内置的安全机制。为了进一步防止LRM制定对策，我们结合了两种动态策略--随机和自适应--来调整密码长度、顺序和组合。对真实世界推理模型（包括DeepSeek-R1、Claude Sonnet和OpenAI GPT-o 4）的广泛实验验证了我们方法的有效性。值得注意的是，SEAL在GPT o 4-mini上的攻击成功率为80.8%，远远超过最先进的基线27.2%。警告：本文包含不恰当、冒犯性和有害内容的示例。



## **19. LLM-BSCVM: An LLM-Based Blockchain Smart Contract Vulnerability Management Framework**

LLM-BSCVC：基于LLM的区块链智能合同漏洞管理框架 cs.CR

10 pages, 8 figures

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17416v1) [paper-pdf](http://arxiv.org/pdf/2505.17416v1)

**Authors**: Yanli Jin, Chunpei Li, Peng Fan, Peng Liu, Xianxian Li, Chen Liu, Wangjie Qiu

**Abstract**: Smart contracts are a key component of the Web 3.0 ecosystem, widely applied in blockchain services and decentralized applications. However, the automated execution feature of smart contracts makes them vulnerable to potential attacks due to inherent flaws, which can lead to severe security risks and financial losses, even threatening the integrity of the entire decentralized finance system. Currently, research on smart contract vulnerabilities has evolved from traditional program analysis methods to deep learning techniques, with the gradual introduction of Large Language Models. However, existing studies mainly focus on vulnerability detection, lacking systematic cause analysis and Vulnerability Repair. To address this gap, we propose LLM-BSCVM, a Large Language Model-based smart contract vulnerability management framework, designed to provide end-to-end vulnerability detection, analysis, repair, and evaluation capabilities for Web 3.0 ecosystem. LLM-BSCVM combines retrieval-augmented generation technology and multi-agent collaboration, introducing a three-stage method of Decompose-Retrieve-Generate. This approach enables smart contract vulnerability management through the collaborative efforts of six intelligent agents, specifically: vulnerability detection, cause analysis, repair suggestion generation, risk assessment, vulnerability repair, and patch evaluation. Experimental results demonstrate that LLM-BSCVM achieves a vulnerability detection accuracy and F1 score exceeding 91\% on benchmark datasets, comparable to the performance of state-of-the-art (SOTA) methods, while reducing the false positive rate from 7.2\% in SOTA methods to 5.1\%, thus enhancing the reliability of vulnerability management. Furthermore, LLM-BSCVM supports continuous security monitoring and governance of smart contracts through a knowledge base hot-swapping dynamic update mechanism.

摘要: 智能合约是Web 3.0生态系统的关键组成部分，广泛应用于区块链服务和去中心化应用程序。然而，智能合约的自动执行功能使其容易受到潜在攻击，这可能导致严重的安全风险和财务损失，甚至威胁到整个分散式金融系统的完整性。目前，对智能合约漏洞的研究已经从传统的程序分析方法发展到深度学习技术，并逐步引入大语言模型。然而，现有的研究主要集中在漏洞检测方面，缺乏系统的原因分析和漏洞修复。为了解决这一差距，我们提出了LLM-BSCVM，这是一个基于大语言模型的智能合约漏洞管理框架，旨在为Web 3.0生态系统提供端到端漏洞检测、分析、修复和评估能力。LLM-BSC虚拟机将检索增强生成技术和多代理协作相结合，引入了分解-分解-生成的三阶段方法。这种方法通过六个智能代理的协作来实现智能合约漏洞管理，具体而言：漏洞检测、原因分析、修复建议生成、风险评估、漏洞修复和补丁评估。实验结果表明，LLM-BSCVC在基准数据集上实现了漏洞检测准确率和F1评分超过91%，与最新技术（SOTA）方法的性能相当，同时将假阳性率从SOTA方法的7.2%降低到5.1%，从而提高了漏洞管理的可靠性。此外，LLM-BSCVC通过知识库热交换动态更新机制支持智能合约的持续安全监控和治理。



## **20. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

X-Transfer攻击：CLIP上的超级可转移对抗攻击 cs.CV

ICML 2025

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.05528v2) [paper-pdf](http://arxiv.org/pdf/2505.05528v2)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.

摘要: 随着对比图像预训练（CLIP）模型越来越多地被用于各种下游任务并集成到大型视觉语言模型（VLM）中，它们对对抗性扰动的敏感性已成为一个关键问题。在这项工作中，我们介绍了\textbf{X-Transfer}，一种新的攻击方法，暴露了CLIP中的一个普遍的对抗性漏洞。X-Transfer生成一个通用对抗扰动（Universal Adversarial Perturbation，UAP），能够欺骗不同样本、任务和域中的各种CLIP编码器和下游VLM。我们将此属性称为\textbf{super transferability}--一个同时实现跨数据、跨域、跨模型和跨任务对抗性可转移性的单一扰动。这是通过\textBF{代理缩放}来实现的，这是我们方法的一个关键创新。与依赖于固定代理模型（扩展计算密集型）的现有方法不同，X-Transfer采用高效的代理扩展策略，可以从大搜索空间中动态选择合适代理的一小子集。广泛的评估表明，X-Transfer的性能显着优于之前最先进的UAP方法，为跨CLIP模型的对抗性可移植性建立了新的基准。该代码可在我们的\href{https：//github.com/HanxunH/XTransferBench}{GitHub存储库}中公开获取。



## **21. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

StealthRank：通过Stealthy提示优化进行LLM排名操纵 cs.IR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2504.05804v2) [paper-pdf](http://arxiv.org/pdf/2504.05804v2)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present $\textbf{StealthRank}$, a novel adversarial attack method that manipulates LLM-driven ranking systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within item or document descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target items while avoiding explicit manipulation traces. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven ranking systems. Our code is publicly available at $\href{https://github.com/Tangyiming205069/controllable-seo}{here}$.

摘要: 将大型语言模型（LLM）集成到信息检索系统中引入了新的攻击表面，特别是对于对抗性排名操纵。我们提出了$\textBF{StealthRank}$，这是一种新型的对抗攻击方法，可以操纵LLM驱动的排名系统，同时保持文本流畅性和隐蔽性。与经常引入可检测异常的现有方法不同，StealthRank采用基于能量的优化框架与Langevin动态相结合来生成StealthRank脚本（SPP）-嵌入在项目或文档描述中的对抗性文本序列，微妙而有效地影响LLM排名机制。我们在多个LLM中评估StealthRank，证明其能够秘密提高目标项目的排名，同时避免显式操纵痕迹。我们的结果表明，StealthRank在有效性和隐蔽性方面始终优于最先进的对抗排名基线，凸显了LLM驱动的排名系统中的关键漏洞。我们的代码可在$\href{https：//github.com/Tangyiming205069/guardable-seo}{here}$公开。



## **22. Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities in MLLM-Powered Mobile GUI Agents**

Hidden Ghost Hand：揭露MLLM支持的移动图形用户界面代理中的后门漏洞 cs.CL

25 pages, 10 figures, 12 Tables

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.14418v2) [paper-pdf](http://arxiv.org/pdf/2505.14418v2)

**Authors**: Pengzhou Cheng, Haowen Hu, Zheng Wu, Zongru Wu, Tianjie Ju, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Graphical user interface (GUI) agents powered by multimodal large language models (MLLMs) have shown greater promise for human-interaction. However, due to the high fine-tuning cost, users often rely on open-source GUI agents or APIs offered by AI providers, which introduces a critical but underexplored supply chain threat: backdoor attacks. In this work, we first unveil that MLLM-powered GUI agents naturally expose multiple interaction-level triggers, such as historical steps, environment states, and task progress. Based on this observation, we introduce AgentGhost, an effective and stealthy framework for red-teaming backdoor attacks. Specifically, we first construct composite triggers by combining goal and interaction levels, allowing GUI agents to unintentionally activate backdoors while ensuring task utility. Then, we formulate backdoor injection as a Min-Max optimization problem that uses supervised contrastive learning to maximize the feature difference across sample classes at the representation space, improving flexibility of the backdoor. Meanwhile, it adopts supervised fine-tuning to minimize the discrepancy between backdoor and clean behavior generation, enhancing effectiveness and utility. Extensive evaluations of various agent models in two established mobile benchmarks show that AgentGhost is effective and generic, with attack accuracy that reaches 99.7\% on three attack objectives, and shows stealthiness with only 1\% utility degradation. Furthermore, we tailor a defense method against AgentGhost that reduces the attack accuracy to 22.1\%. Our code is available at \texttt{anonymous}.

摘要: 由多模式大型语言模型（MLLM）支持的图形用户界面（图形用户界面）代理在人际交互方面表现出了更大的前景。然而，由于微调成本很高，用户通常依赖人工智能提供商提供的开源图形界面代理或API，这引入了一个关键但未充分开发的供应链威胁：后门攻击。在这项工作中，我们首先揭示了基于MLLM的图形用户界面代理自然暴露多个交互级触发器，例如历史步骤、环境状态和任务进度。基于这一观察，我们引入了AgentGhost，这是一个用于红色团队后门攻击的有效且隐蔽的框架。具体来说，我们首先通过结合目标和交互级别来构建复合触发器，允许图形用户界面代理无意中激活后门，同时确保任务实用性。然后，我们将后门注入制定为Min-Max优化问题，该问题使用监督对比学习来最大化表示空间中样本类之间的特征差异，从而提高后门的灵活性。同时，它采用监督式微调，以最大限度地减少后门和干净行为生成之间的差异，提高有效性和实用性。对两个已建立的移动基准测试中各种代理模型的广泛评估表明，AgentGhost有效且通用，在三个攻击目标上的攻击准确率达到99.7%，并且表现出隐蔽性，仅使用1%的效用下降。此外，我们针对AgentGhost定制了一种防御方法，将攻击准确率降低至22.1%。我们的代码可在\textttt {anonymous}上获取。



## **23. Advancing Security with Digital Twins: A Comprehensive Survey**

通过数字双胞胎推进安全性：全面调查 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17310v1) [paper-pdf](http://arxiv.org/pdf/2505.17310v1)

**Authors**: Blessing Airehenbuwa, Touseef Hasan, Souvika Sarkar, Ujjwal Guin

**Abstract**: The proliferation of electronic devices has greatly transformed every aspect of human life, such as communication, healthcare, transportation, and energy. Unfortunately, the global electronics supply chain is vulnerable to various attacks, including piracy of intellectual properties, tampering, counterfeiting, information leakage, side-channel, and fault injection attacks, due to the complex nature of electronic products and vulnerabilities present in them. Although numerous solutions have been proposed to address these threats, significant gaps remain, particularly in providing scalable and comprehensive protection against emerging attacks. Digital twin, a dynamic virtual replica of a physical system, has emerged as a promising solution to address these issues by providing backward traceability, end-to-end visibility, and continuous verification of component integrity and behavior. In this paper, we present a comprehensive survey of the application of digital twins based on their functional role and application domains. We comprehensively present recent digital twin-based security implementations, including their role in cyber-physical systems, Internet of Things, and cryptographic systems, detection of counterfeit electronics, intrusion detection, fault injection, and side-channel leakage. To the best of our knowledge, it is the first study to consolidate these security use cases into a unified reference. The paper also explores the integration of large language models with digital twins for enhanced security and discusses current challenges, solutions, and future research directions.

摘要: 电子设备的激增极大地改变了人类生活的各个方面，例如通信、医疗保健、交通和能源。不幸的是，由于电子产品的复杂性及其存在的漏洞，全球电子供应链很容易受到各种攻击，包括知识产权盗版、篡改、假冒、信息泄露、侧通道和故障注入攻击。尽管已经提出了许多解决方案来解决这些威胁，但仍然存在巨大差距，特别是在针对新出现的攻击提供可扩展和全面的保护方面。Digital twin是物理系统的动态虚拟副本，通过提供向后可追溯性、端到端可见性以及组件完整性和行为的持续验证，已成为解决这些问题的一种有前途的解决方案。本文根据数字双胞胎的功能角色和应用领域对数字双胞胎的应用进行了全面的调查。我们全面介绍了最近的数字孪生安全实施，包括它们在网络物理系统、物联网和加密系统中的作用、假冒电子产品的检测、入侵检测、故障注入和侧通道泄漏。据我们所知，这是第一项将这些安全用例整合为统一参考的研究。本文还探讨了大型语言模型与数字双胞胎的集成以增强安全性，并讨论了当前的挑战、解决方案和未来的研究方向。



## **24. TrustRAG: Enhancing Robustness and Trustworthiness in Retrieval-Augmented Generation**

TrustRAG：增强检索增强一代的鲁棒性和可信度 cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2501.00879v3) [paper-pdf](http://arxiv.org/pdf/2501.00879v3)

**Authors**: Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen, Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, Emine Yilmaz

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by integrating external knowledge sources, enabling more accurate and contextually relevant responses tailored to user queries. These systems, however, remain susceptible to corpus poisoning attacks, which can severely impair the performance of LLMs. To address this challenge, we propose TrustRAG, a robust framework that systematically filters malicious and irrelevant content before it is retrieved for generation. Our approach employs a two-stage defense mechanism. The first stage implements a cluster filtering strategy to detect potential attack patterns. The second stage employs a self-assessment process that harnesses the internal capabilities of LLMs to detect malicious documents and resolve inconsistencies. TrustRAG provides a plug-and-play, training-free module that integrates seamlessly with any open- or closed-source language model. Extensive experiments demonstrate that TrustRAG delivers substantial improvements in retrieval accuracy, efficiency, and attack resistance.

摘要: 检索增强生成（RAG）通过集成外部知识源来增强大型语言模型（LLM），从而实现针对用户查询量身定制的更准确且上下文相关的响应。然而，这些系统仍然容易受到主体中毒攻击，这可能会严重损害LLM的性能。为了应对这一挑战，我们提出了TrustRAG，这是一个强大的框架，可以在检索恶意和不相关内容以生成之前系统地过滤恶意和不相关内容。我们的方法采用两阶段防御机制。第一阶段实施集群过滤策略来检测潜在的攻击模式。第二阶段采用自我评估流程，利用LLM的内部功能来检测恶意文档并解决不一致问题。TrustRAG提供了一个即插即用、免培训模块，可以与任何开源或闭源语言模型无缝集成。大量实验表明，TrustRAG在检索准确性、效率和抗攻击性方面提供了重大改进。



## **25. Invisible Prompts, Visible Threats: Malicious Font Injection in External Resources for Large Language Models**

看不见的警告，可见的威胁：大型语言模型的外部资源中的恶意字体注入 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16957v1) [paper-pdf](http://arxiv.org/pdf/2505.16957v1)

**Authors**: Junjie Xiong, Changjia Zhu, Shuhang Lin, Chong Zhang, Yongfeng Zhang, Yao Liu, Lingyao Li

**Abstract**: Large Language Models (LLMs) are increasingly equipped with capabilities of real-time web search and integrated with protocols like Model Context Protocol (MCP). This extension could introduce new security vulnerabilities. We present a systematic investigation of LLM vulnerabilities to hidden adversarial prompts through malicious font injection in external resources like webpages, where attackers manipulate code-to-glyph mapping to inject deceptive content which are invisible to users. We evaluate two critical attack scenarios: (1) "malicious content relay" and (2) "sensitive data leakage" through MCP-enabled tools. Our experiments reveal that indirect prompts with injected malicious font can bypass LLM safety mechanisms through external resources, achieving varying success rates based on data sensitivity and prompt design. Our research underscores the urgent need for enhanced security measures in LLM deployments when processing external content.

摘要: 大型语言模型（LLM）越来越多地配备实时网络搜索功能，并与模型上下文协议（HCP）等协议集成。此扩展可能会引入新的安全漏洞。我们对通过在网页等外部资源中恶意字体注入来隐藏对抗提示的LLM漏洞进行了系统性调查，其中攻击者操纵代码到收件箱的映射来注入用户不可见的欺骗性内容。我们评估了两种关键攻击场景：（1）“恶意内容中继”和（2）通过支持MVP的工具“敏感数据泄露”。我们的实验表明，注入恶意字体的间接提示可以通过外部资源绕过LLM安全机制，根据数据敏感性和提示设计实现不同的成功率。我们的研究强调了处理外部内容时LLM部署中迫切需要增强的安全措施。



## **26. MixAT: Combining Continuous and Discrete Adversarial Training for LLMs**

MixAT：结合LLM的连续和离散对抗训练 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16947v1) [paper-pdf](http://arxiv.org/pdf/2505.16947v1)

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.

摘要: 尽管最近在大型语言模型（LLM）的安全性和一致性方面做出了努力，但当前对前沿LLM的对抗性攻击仍然能够持续地迫使有害的世代。尽管对抗训练已得到广泛研究，并被证明可以显着提高传统机器学习模型的鲁棒性，但其在LLM背景下的优点和缺点却知之甚少。具体来说，虽然现有的离散对抗攻击可以有效地产生有害内容，但用具体的对抗提示训练LLM通常计算成本高昂，导致依赖于持续的放松。由于这些松弛不对应于离散输入令牌，因此此类潜在训练方法通常使模型容易受到一系列不同的离散攻击。在这项工作中，我们的目标是通过引入MixAT来弥合这一差距，MixAT是一种新颖的方法，在训练期间结合了更强的离散攻击和更快的连续攻击。我们对MixAT进行了广泛的最先进攻击，提出了至少一次攻击成功率（ALO-ASB）指标来捕捉模型的最坏情况漏洞。我们表明，与之前的防御（ALO-ASB> 50%）相比，MixAT实现了更好的鲁棒性（ALO-ASB < 20%），同时保持与基于连续松弛的方法相当的运行时间。我们进一步分析了现实部署环境中的MixAT，探索聊天模板、量化、低等级适配器和温度如何影响对抗训练和评估，从而揭示了当前方法中的其他盲点。我们的结果表明，MixAT的离散-连续防御以最小的计算负担提供了原则性且卓越的鲁棒性-准确性权衡，凸显了其构建更安全的LLM的承诺。我们在https://github.com/insait-institute/MixAT上提供我们的代码和模型。



## **27. Backdoor Cleaning without External Guidance in MLLM Fine-tuning**

MLLM微调中未经外部指导的后门清理 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16916v1) [paper-pdf](http://arxiv.org/pdf/2505.16916v1)

**Authors**: Xuankun Rong, Wenke Huang, Jian Liang, Jinhe Bi, Xun Xiao, Yiming Li, Bo Du, Mang Ye

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly deployed in fine-tuning-as-a-service (FTaaS) settings, where user-submitted datasets adapt general-purpose models to downstream tasks. This flexibility, however, introduces serious security risks, as malicious fine-tuning can implant backdoors into MLLMs with minimal effort. In this paper, we observe that backdoor triggers systematically disrupt cross-modal processing by causing abnormal attention concentration on non-semantic regions--a phenomenon we term attention collapse. Based on this insight, we propose Believe Your Eyes (BYE), a data filtering framework that leverages attention entropy patterns as self-supervised signals to identify and filter backdoor samples. BYE operates via a three-stage pipeline: (1) extracting attention maps using the fine-tuned model, (2) computing entropy scores and profiling sensitive layers via bimodal separation, and (3) performing unsupervised clustering to remove suspicious samples. Unlike prior defenses, BYE equires no clean supervision, auxiliary labels, or model modifications. Extensive experiments across various datasets, models, and diverse trigger types validate BYE's effectiveness: it achieves near-zero attack success rates while maintaining clean-task performance, offering a robust and generalizable solution against backdoor threats in MLLMs.

摘要: 多模式大型语言模型（MLLM）越来越多地部署在微调即服务（FTSaaS）设置中，其中用户提交的数据集将通用模型适应下游任务。然而，这种灵活性会带来严重的安全风险，因为恶意微调可以以最少的努力将后门植入MLLM中。在本文中，我们观察到后门触发器通过导致非语义区域的异常注意集中来系统性地扰乱跨模式处理--我们将这种现象称为注意力崩溃。基于这一见解，我们提出了相信你的眼睛（BYE），这是一种数据过滤框架，利用注意力熵模式作为自我监督信号来识别和过滤后门样本。BYE通过三阶段流水线运行：（1）使用微调模型提取注意力图，（2）通过双峰分离计算熵分数并分析敏感层，以及（3）执行无监督集群以删除可疑样本。与之前的防御不同，BYE不提供干净的监督、辅助标签或型号修改。跨各种数据集、模型和不同触发类型的广泛实验验证了BYE的有效性：它实现了接近零的攻击成功率，同时保持干净任务性能，提供了针对MLLM中后门威胁的强大且可推广的解决方案。



## **28. CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework**

CAIN：通过两阶段恶意系统提示生成和精炼框架劫持LLM与人类对话 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16888v1) [paper-pdf](http://arxiv.org/pdf/2505.16888v1)

**Authors**: Viet Pham, Thai Le

**Abstract**: Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.

摘要: 大型语言模型（LLM）先进了许多应用程序，但也容易受到对抗攻击。在这项工作中，我们引入了一种新颖的安全威胁：通过操纵LLM的系统提示来劫持人工智能与人类的对话，以仅对特定目标问题（例如，“我应该投票给谁美国总统？”，“新冠疫苗安全吗？”），同时对他人表现友善。这种攻击是有害的，因为它可以使恶意行为者通过在线传播有害但看起来友善的系统提示来进行大规模信息操纵。为了演示此类攻击，我们开发了CAIN，这是一种算法，可以在黑匣子设置中或无需访问LLM参数的情况下自动策划此类有害系统提示特定目标问题。在开源和商业LLM上进行评估，CAIN表现出显着的对抗影响。在无针对性攻击或迫使LLM输出错误答案中，CAIN对目标问题实现了高达40%的F1降级，同时对良性输入保持高准确性。对于有针对性的攻击或迫使LLM输出特定的有害答案，CAIN在这些有针对性的回答上获得了超过70%的F1分数，而对良性问题的影响最小。我们的结果凸显了对增强稳健性措施的迫切需要，以保障LLM在现实世界应用中的完整性和安全性。所有源代码都将公开。



## **29. Safe RLHF-V: Safe Reinforcement Learning from Multi-modal Human Feedback**

Safe RLHF-V：基于多模态人类反馈的安全强化学习 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2503.17682v2) [paper-pdf](http://arxiv.org/pdf/2503.17682v2)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Conghui Zhang, Han Zhu, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Yida Tang, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are essential for building general-purpose AI assistants; however, they pose increasing safety risks. How can we ensure safety alignment of MLLMs to prevent undesired behaviors? Going further, it is critical to explore how to fine-tune MLLMs to preserve capabilities while meeting safety constraints. Fundamentally, this challenge can be formulated as a min-max optimization problem. However, existing datasets have not yet disentangled single preference signals into explicit safety constraints, hindering systematic investigation in this direction. Moreover, it remains an open question whether such constraints can be effectively incorporated into the optimization process for multi-modal models. In this work, we present the first exploration of the Safe RLHF-V -- the first multimodal safety alignment framework. The framework consists of: $\mathbf{(I)}$ BeaverTails-V, the first open-source dataset featuring dual preference annotations for helpfulness and safety, supplemented with multi-level safety labels (minor, moderate, severe); $\mathbf{(II)}$ Beaver-Guard-V, a multi-level guardrail system to proactively defend against unsafe queries and adversarial attacks. Applying the guard model over five rounds of filtering and regeneration significantly enhances the precursor model's overall safety by an average of 40.9%. $\mathbf{(III)}$ Based on dual preference, we initiate the first exploration of multi-modal safety alignment within a constrained optimization. Experimental results demonstrate that Safe RLHF effectively improves both model helpfulness and safety. Specifically, Safe RLHF-V enhances model safety by 34.2% and helpfulness by 34.3%.

摘要: 多模式大型语言模型（MLLM）对于构建通用人工智能助手至关重要;然而，它们带来了越来越大的安全风险。我们如何确保MLLM的安全一致以防止不良行为？进一步说，探索如何微调MLLM以在满足安全限制的同时保留功能至关重要。从根本上讲，这个挑战可以被描述为一个最小-最大优化问题。然而，现有的数据集尚未将单一偏好信号分解为明确的安全约束，从而阻碍了这方面的系统性研究。此外，这些约束是否可以有效地纳入多模式模型的优化过程仍然是一个悬而未决的问题。在这项工作中，我们首次探索Safe RLHF-V --第一个多模式安全对齐框架。该框架包括：$\mathBF{（I）}$ BeaverTails-V，第一个开源数据集，具有帮助性和安全性的双重偏好注释，并辅之以多级别安全标签（轻微、中度、严重）; $\mathBF{（II）}$ Beaver-Guard-V，一个多级别护栏系统，用于主动防御不安全的查询和对抗性攻击。经过五轮过滤和再生应用防护模型，前体模型的整体安全性平均显着提高了40.9%。$\mathBF{（III）}$基于双重偏好，我们在约束优化中启动了多模式安全对齐的首次探索。实验结果表明，Safe RL HF有效提高了模型的帮助性和安全性。具体而言，Safe RLHF-V将模型的安全性提高了34.2%，帮助性提高了34.3%。



## **30. Accidental Misalignment: Fine-Tuning Language Models Induces Unexpected Vulnerability**

意外失调：微调语言模型会引发意外漏洞 cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16789v1) [paper-pdf](http://arxiv.org/pdf/2505.16789v1)

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Abstract**: As large language models gain popularity, their vulnerability to adversarial attacks remains a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Misalignment, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity within our experimental datasets. We then evaluate the adversarial performance of these fine-tuned models and assess how dataset factors correlate with attack success rates. Lastly, we explore potential causal links, offering new insights into adversarial defense strategies and highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_misalignment.

摘要: 随着大型语言模型越来越受欢迎，它们对对抗攻击的脆弱性仍然是一个主要问题。虽然通常使用特定领域数据集的微调模型来提高模型性能，但它可能会在基础模型中引入漏洞。在这项工作中，我们调查了意外失准，即微调数据特征引起的意外漏洞。我们首先确定潜在的相关因素，如语言特征，语义相似性和毒性在我们的实验数据集。然后，我们评估这些微调模型的对抗性能，并评估数据集因素与攻击成功率的相关性。最后，我们探索了潜在的因果关系，为对抗性防御策略提供了新的见解，并强调了数据集设计在保持模型对齐方面的关键作用。我们的代码可在https://github.com/psyonp/accidental_misalignment上获取。



## **31. When Safety Detectors Aren't Enough: A Stealthy and Effective Jailbreak Attack on LLMs via Steganographic Techniques**

当安全检测器还不够时：通过隐写技术对LLM进行秘密有效的越狱攻击 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16765v1) [paper-pdf](http://arxiv.org/pdf/2505.16765v1)

**Authors**: Jianing Geng, Biao Yi, Zekun Fei, Tongxi Wu, Lihai Nie, Zheli Liu

**Abstract**: Jailbreak attacks pose a serious threat to large language models (LLMs) by bypassing built-in safety mechanisms and leading to harmful outputs. Studying these attacks is crucial for identifying vulnerabilities and improving model security. This paper presents a systematic survey of jailbreak methods from the novel perspective of stealth. We find that existing attacks struggle to simultaneously achieve toxic stealth (concealing toxic content) and linguistic stealth (maintaining linguistic naturalness). Motivated by this, we propose StegoAttack, a fully stealthy jailbreak attack that uses steganography to hide the harmful query within benign, semantically coherent text. The attack then prompts the LLM to extract the hidden query and respond in an encrypted manner. This approach effectively hides malicious intent while preserving naturalness, allowing it to evade both built-in and external safety mechanisms. We evaluate StegoAttack on four safety-aligned LLMs from major providers, benchmarking against eight state-of-the-art methods. StegoAttack achieves an average attack success rate (ASR) of 92.00%, outperforming the strongest baseline by 11.0%. Its ASR drops by less than 1% even under external detection (e.g., Llama Guard). Moreover, it attains the optimal comprehensive scores on stealth detection metrics, demonstrating both high efficacy and exceptional stealth capabilities. The code is available at https://anonymous.4open.science/r/StegoAttack-Jail66

摘要: 越狱攻击绕过内置安全机制并导致有害输出，对大型语言模型（LLM）构成严重威胁。研究这些攻击对于识别漏洞和提高模型安全性至关重要。本文从隐身的新颖角度对越狱方法进行了系统的概述。我们发现现有的攻击很难同时实现有毒隐形（隐藏有毒内容）和语言隐形（保持语言自然性）。出于此动机，我们提出了StegoAttack，这是一种完全隐蔽的越狱攻击，使用隐写术将有害查询隐藏在良性、语义连贯的文本中。然后，攻击会促使LLM提取隐藏的查询并以加密方式响应。这种方法有效地隐藏恶意意图，同时保持自然性，使其能够逃避内置和外部安全机制。我们评估StegoAttack对四个安全对齐的LLM从主要供应商，基准对八个国家的最先进的方法。StegoAttack的平均攻击成功率（ASR）为92.00%，比最强基线高出11.0%。即使在外部检测下，其ASR也下降不到1%（例如，Llama Guard）。此外，它还获得了最佳的隐身检测指标综合评分，展示了高功效和出色的隐身能力。该代码可在https://anonymous.4open.science/r/StegoAttack-Jail66上获取



## **32. BitHydra: Towards Bit-flip Inference Cost Attack against Large Language Models**

BitHydra：针对大型语言模型的位翻转推理成本攻击 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16670v1) [paper-pdf](http://arxiv.org/pdf/2505.16670v1)

**Authors**: Xiaobei Yan, Yiming Li, Zhaoxin Fan, Han Qiu, Tianwei Zhang

**Abstract**: Large language models (LLMs) have shown impressive capabilities across a wide range of applications, but their ever-increasing size and resource demands make them vulnerable to inference cost attacks, where attackers induce victim LLMs to generate the longest possible output content. In this paper, we revisit existing inference cost attacks and reveal that these methods can hardly produce large-scale malicious effects since they are self-targeting, where attackers are also the users and therefore have to execute attacks solely through the inputs, whose generated content will be charged by LLMs and can only directly influence themselves. Motivated by these findings, this paper introduces a new type of inference cost attacks (dubbed 'bit-flip inference cost attack') that target the victim model itself rather than its inputs. Specifically, we design a simple yet effective method (dubbed 'BitHydra') to effectively flip critical bits of model parameters. This process is guided by a loss function designed to suppress <EOS> token's probability with an efficient critical bit search algorithm, thus explicitly defining the attack objective and enabling effective optimization. We evaluate our method on 11 LLMs ranging from 1.5B to 14B parameters under both int8 and float16 settings. Experimental results demonstrate that with just 4 search samples and as few as 3 bit flips, BitHydra can force 100% of test prompts to reach the maximum generation length (e.g., 2048 tokens) on representative LLMs such as LLaMA3, highlighting its efficiency, scalability, and strong transferability across unseen inputs.

摘要: 大型语言模型（LLM）在广泛的应用程序中表现出令人印象深刻的能力，但其不断增加的规模和资源需求使它们容易受到推理成本攻击，攻击者诱导受害者LLM生成尽可能长的输出内容。在本文中，我们回顾了现有的推理成本攻击，并揭示了这些方法很难产生大规模的恶意影响，因为它们是自瞄准的，攻击者也是用户，因此必须仅通过输入来执行攻击，其生成的内容将由LLM收费并且只能直接影响自己。受这些发现的启发，本文引入了一种新型的推理成本攻击（称为“位翻转推理成本攻击”），其目标是受害者模型本身，而不是其输入。具体来说，我们设计了一种简单而有效的方法（称为“BitHydra”）来有效地翻转模型参数的关键部分。该过程由损失函数指导，该函数旨在<EOS>通过高效的关键位搜索算法抑制令牌的概率，从而明确定义攻击目标并实现有效的优化。我们在int8和float 16设置下对11个LLM（参数范围从1.5B到14B）上评估了我们的方法。实验结果表明，只需4个搜索样本和少至3位翻转，BitHydra就可以强制100%的测试提示达到最大生成长度（例如，2048个令牌），突出了其效率，可扩展性和跨看不见的输入的强大可转移性。



## **33. Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models**

分而治之：击败多模式大型语言模型的混合策略 cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2412.16555v2) [paper-pdf](http://arxiv.org/pdf/2412.16555v2)

**Authors**: Yanxu Mao, Peipei Liu, Tiehan Cui, Zhaoteng Yan, Congying Liu, Datao You

**Abstract**: Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive dataset for multimodal jailbreaking research: TriJail, which includes jailbreak prompts for all three modalities. Experiments on the TriJail dataset and the benchmark dataset AdvBench, conducted on 13 popular LLMs, demonstrate advanced attack success rates and significant reduction in time overhead.

摘要: 大型语言模型（LLM）因其强大的推理、理解和生成能力而广泛应用于社会各个领域。然而，与这些模型相关的安全问题正变得日益严重。越狱攻击作为检测LLM漏洞的重要方法，已被研究人员探索，他们试图通过各种攻击方法诱导这些模型生成有害内容。然而，现有的越狱方法面临着许多局限性，例如过多的查询次数、越狱模式的覆盖范围有限、攻击成功率低以及评估方法简单化。为了克服这些限制，本文提出了一种多模式越狱方法：JMLLM。该方法集成了多种策略，以跨文本、视觉和听觉方式执行全面的越狱攻击。此外，我们还为多模式越狱研究提供了一个新的全面数据集：TriJail，其中包括所有三种模式的越狱提示。在TriJail数据集和基准数据集AdvBench上进行的实验在13个流行的LLM上进行，展示了先进的攻击成功率和显着减少的时间成本。



## **34. From Evaluation to Defense: Advancing Safety in Video Large Language Models**

从评估到防御：提高视频大型语言模型的安全性 cs.CV

49 pages, 12 figures, 17 tables

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16643v1) [paper-pdf](http://arxiv.org/pdf/2505.16643v1)

**Authors**: Yiwei Sun, Peiqi Jiang, Chuanbin Liu, Luohao Lin, Zhiying Lu, Hongtao Xie

**Abstract**: While the safety risks of image-based large language models have been extensively studied, their video-based counterparts (Video LLMs) remain critically under-examined. To systematically study this problem, we introduce \textbf{VideoSafetyBench (VSB-77k) - the first large-scale, culturally diverse benchmark for Video LLM safety}, which compromises 77,646 video-query pairs and spans 19 principal risk categories across 10 language communities. \textit{We reveal that integrating video modality degrades safety performance by an average of 42.3\%, exposing systemic risks in multimodal attack exploitation.} To address this vulnerability, we propose \textbf{VideoSafety-R1}, a dual-stage framework achieving unprecedented safety gains through two innovations: (1) Alarm Token-Guided Safety Fine-Tuning (AT-SFT) injects learnable alarm tokens into visual and textual sequences, enabling explicit harm perception across modalities via multitask objectives. (2) Then, Safety-Guided GRPO enhances defensive reasoning through dynamic policy optimization with rule-based rewards derived from dual-modality verification. These components synergize to shift safety alignment from passive harm recognition to active reasoning. The resulting framework achieves a 65.1\% improvement on VSB-Eval-HH, and improves by 59.1\%, 44.3\%, and 15.0\% on the image safety datasets MMBench, VLGuard, and FigStep, respectively. \textit{Our codes are available in the supplementary materials.} \textcolor{red}{Warning: This paper contains examples of harmful language and videos, and reader discretion is recommended.}

摘要: 虽然基于图像的大型语言模型的安全风险已经得到了广泛研究，但其基于视频的对应模型（视频LLM）仍然受到严重不足的审查。为了系统性地研究这个问题，我们引入了\textBF{VideoSafetyBench（TSB-77 k）-第一个大规模、文化多样性的视频LLM安全基准}，它包含77，646个视频查询对，涵盖10个语言社区的19个主要风险类别。\textit{我们发现，集成视频模式会使安全性能平均降低42.3%，暴露了多模式攻击利用中的系统性风险。}为了解决这个漏洞，我们提出了\textBF{VideoSafety-R1}，这是一个双阶段框架，通过两项创新实现前所未有的安全收益：（1）警报令牌引导安全微调（AT-SFT）将可学习的警报令牌注入视觉和文本序列中，通过多任务目标实现跨模式的明确伤害感知。(2)然后，安全引导的GRPO通过动态策略优化和双模式验证中的基于规则的奖励来增强防御推理。这些组件协同作用，将安全调整从被动伤害识别转变为主动推理。最终的框架在VSB-Eval-HH上实现了65.1%的改进，在图像安全数据集MMBench、VLGuard和FigStep上分别提高了59.1%、44.3%和15.0%。\texttit {我们的代码可在补充材料中找到。} \textColor{red}{警告：本文包含有害语言和视频的示例，建议读者自行决定。}



## **35. BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization**

BadVLA：通过解耦优化实现对视觉-语言-动作模型的后门攻击 cs.CR

19 pages, 12 figures, 6 tables

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16640v1) [paper-pdf](http://arxiv.org/pdf/2505.16640v1)

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Hechang Wang, Pan Zhou, Lichao Sun

**Abstract**: Vision-Language-Action (VLA) models have advanced robotic control by enabling end-to-end decision-making directly from multimodal inputs. However, their tightly coupled architectures expose novel security vulnerabilities. Unlike traditional adversarial perturbations, backdoor attacks represent a stealthier, persistent, and practically significant threat-particularly under the emerging Training-as-a-Service paradigm-but remain largely unexplored in the context of VLA models. To address this gap, we propose BadVLA, a backdoor attack method based on Objective-Decoupled Optimization, which for the first time exposes the backdoor vulnerabilities of VLA models. Specifically, it consists of a two-stage process: (1) explicit feature-space separation to isolate trigger representations from benign inputs, and (2) conditional control deviations that activate only in the presence of the trigger, while preserving clean-task performance. Empirical results on multiple VLA benchmarks demonstrate that BadVLA consistently achieves near-100% attack success rates with minimal impact on clean task accuracy. Further analyses confirm its robustness against common input perturbations, task transfers, and model fine-tuning, underscoring critical security vulnerabilities in current VLA deployments. Our work offers the first systematic investigation of backdoor vulnerabilities in VLA models, highlighting an urgent need for secure and trustworthy embodied model design practices. We have released the project page at https://badvla-project.github.io/.

摘要: 视觉-语言-动作（VLA）模型通过直接从多模式输入进行端到端决策，实现了先进的机器人控制。然而，它们的紧密耦合架构暴露了新型安全漏洞。与传统的对抗性扰动不同，后门攻击代表了一种更隐蔽、持久且实际上重大的威胁--特别是在新兴的“服务培训”范式下--但在VLA模型的背景下，它在很大程度上尚未被探索。为了弥补这一差距，我们提出了BadVLA，这是一种基于Inbox-Decoupled优化的后门攻击方法，首次暴露了VLA模型的后门漏洞。具体来说，它由两阶段过程组成：（1）显式特征空间分离，以将触发器表示与良性输入隔离，以及（2）仅在触发器存在时激活的条件控制偏差，同时保持干净任务性能。多个VLA基准的经验结果表明，BadVLA始终实现接近100%的攻击成功率，对干净任务准确性的影响最小。进一步的分析证实了它对常见输入扰动、任务传输和模型微调的稳健性，凸显了当前VLA部署中的关键安全漏洞。我们的工作首次对VLA模型中的后门漏洞进行了系统性调查，凸显了对安全且值得信赖的具体模型设计实践的迫切需求。我们已在https://badvla-project.github.io/上发布了项目页面。



## **36. CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning**

CTRAP：嵌入崩溃陷阱以保护大型语言模型免受有害的微调 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16559v1) [paper-pdf](http://arxiv.org/pdf/2505.16559v1)

**Authors**: Biao Yi, Tiansheng Huang, Baolei Zhang, Tong Li, Lihai Nie, Zheli Liu, Li Shen

**Abstract**: Fine-tuning-as-a-service, while commercially successful for Large Language Model (LLM) providers, exposes models to harmful fine-tuning attacks. As a widely explored defense paradigm against such attacks, unlearning attempts to remove malicious knowledge from LLMs, thereby essentially preventing them from being used to perform malicious tasks. However, we highlight a critical flaw: the powerful general adaptability of LLMs allows them to easily bypass selective unlearning by rapidly relearning or repurposing their capabilities for harmful tasks. To address this fundamental limitation, we propose a paradigm shift: instead of selective removal, we advocate for inducing model collapse--effectively forcing the model to "unlearn everything"--specifically in response to updates characteristic of malicious adaptation. This collapse directly neutralizes the very general capabilities that attackers exploit, tackling the core issue unaddressed by selective unlearning. We introduce the Collapse Trap (CTRAP) as a practical mechanism to implement this concept conditionally. Embedded during alignment, CTRAP pre-configures the model's reaction to subsequent fine-tuning dynamics. If updates during fine-tuning constitute a persistent attempt to reverse safety alignment, the pre-configured trap triggers a progressive degradation of the model's core language modeling abilities, ultimately rendering it inert and useless for the attacker. Crucially, this collapse mechanism remains dormant during benign fine-tuning, ensuring the model's utility and general capabilities are preserved for legitimate users. Extensive empirical results demonstrate that CTRAP effectively counters harmful fine-tuning risks across various LLMs and attack settings, while maintaining high performance in benign scenarios. Our code is available at https://anonymous.4open.science/r/CTRAP.

摘要: 微调即服务虽然对于大型语言模型（LLM）提供商来说在商业上取得了成功，但会使模型暴露于有害的微调攻击之下。作为一种广泛探索的针对此类攻击的防御范式，取消学习尝试从LLM中删除恶意知识，从而从本质上防止它们被用来执行恶意任务。然而，我们强调了一个关键缺陷：LLM强大的一般适应性使它们能够通过快速重新学习或重新利用其能力来完成有害任务来轻松绕过选择性取消学习。为了解决这个根本限制，我们提出了一种范式转变：我们主张诱导模型崩溃，而不是选择性删除--有效地迫使模型“忘记一切”--特别是为了响应恶意适应的更新。这种崩溃直接抵消了攻击者利用的非常普遍的能力，解决了选择性取消学习未解决的核心问题。我们引入崩溃陷阱（CTRAP）作为有条件地实现这一概念的实用机制。CTRAP嵌入在对齐过程中，预配置模型对后续微调动态的反应。如果微调期间的更新构成了扭转安全对齐的持续尝试，那么预配置的陷阱就会引发模型核心语言建模能力的逐渐退化，最终使其对攻击者变得惰性和无用。至关重要的是，这种崩溃机制在良性微调期间保持休眠状态，确保为合法用户保留模型的实用性和通用功能。广泛的实证结果表明，CTRAP可以有效地应对各种LLM和攻击设置中的有害微调风险，同时在良性场景中保持高性能。我们的代码可在https://anonymous.4open.science/r/CTRAP上获取。



## **37. Implicit Jailbreak Attacks via Cross-Modal Information Concealment on Vision-Language Models**

通过视觉语言模型的跨模式信息隐藏进行隐性越狱攻击 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16446v1) [paper-pdf](http://arxiv.org/pdf/2505.16446v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Multimodal large language models (MLLMs) enable powerful cross-modal reasoning capabilities. However, the expanded input space introduces new attack surfaces. Previous jailbreak attacks often inject malicious instructions from text into less aligned modalities, such as vision. As MLLMs increasingly incorporate cross-modal consistency and alignment mechanisms, such explicit attacks become easier to detect and block. In this work, we propose a novel implicit jailbreak framework termed IJA that stealthily embeds malicious instructions into images via least significant bit steganography and couples them with seemingly benign, image-related textual prompts. To further enhance attack effectiveness across diverse MLLMs, we incorporate adversarial suffixes generated by a surrogate model and introduce a template optimization module that iteratively refines both the prompt and embedding based on model feedback. On commercial models like GPT-4o and Gemini-1.5 Pro, our method achieves attack success rates of over 90% using an average of only 3 queries.

摘要: 多模式大型语言模型（MLLM）实现强大的跨模式推理能力。然而，扩展的输入空间引入了新的攻击面。之前的越狱攻击经常将文本中的恶意指令注入到不一致的模式中，例如视觉。随着MLLM越来越多地结合跨模式一致性和对齐机制，此类显式攻击变得更容易检测和阻止。在这项工作中，我们提出了一种名为IJA的新型隐式越狱框架，该框架通过最低有效位隐写术将恶意指令秘密地嵌入到图像中，并将其与看似良性的图像相关文本提示相结合。为了进一步增强不同MLLM之间的攻击有效性，我们结合了代理模型生成的对抗性后缀，并引入了模板优化模块，该模块根据模型反馈迭代地细化提示和嵌入。在GPT-4 o和Gemini-1.5 Pro等商业型号上，我们的方法平均只需3个查询即可实现超过90%的攻击成功率。



## **38. MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming**

MTSA：通过多轮红队进行LLM的多圈安全对准 cs.CR

19 pages,6 figures,ACL2025

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17147v1) [paper-pdf](http://arxiv.org/pdf/2505.17147v1)

**Authors**: Weiyang Guo, Jing Li, Wenya Wang, YU LI, Daojing He, Jun Yu, Min Zhang

**Abstract**: The proliferation of jailbreak attacks against large language models (LLMs) highlights the need for robust security measures. However, in multi-round dialogues, malicious intentions may be hidden in interactions, leading LLMs to be more prone to produce harmful responses. In this paper, we propose the \textbf{M}ulti-\textbf{T}urn \textbf{S}afety \textbf{A}lignment (\ourapproach) framework, to address the challenge of securing LLMs in multi-round interactions. It consists of two stages: In the thought-guided attack learning stage, the red-team model learns about thought-guided multi-round jailbreak attacks to generate adversarial prompts. In the adversarial iterative optimization stage, the red-team model and the target model continuously improve their respective capabilities in interaction. Furthermore, we introduce a multi-turn reinforcement learning algorithm based on future rewards to enhance the robustness of safety alignment. Experimental results show that the red-team model exhibits state-of-the-art attack capabilities, while the target model significantly improves its performance on safety benchmarks.

摘要: 针对大型语言模型（LLM）的越狱攻击的激增凸显了对强有力安全措施的必要性。然而，在多轮对话中，恶意意图可能隐藏在互动中，导致LLM更容易产生有害响应。在本文中，我们提出了\textBF{M}ulti-\textBF{T}urn \textBF{S} ajax\textBF{A} lignation（\ourapproach）框架，以解决在多轮交互中保护LLM的挑战。它由两个阶段组成：在思想引导的攻击学习阶段，红队模型学习思想引导的多轮越狱攻击以生成对抗提示。在对抗迭代优化阶段，红队模型和目标模型不断提高各自的交互能力。此外，我们引入了基于未来回报的多轮强化学习算法，以增强安全对齐的鲁棒性。实验结果表明，红队模型展现出最先进的攻击能力，而目标模型在安全基准方面的性能显着提高。



## **39. Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems**

针对基于R1的检索增强生成系统的思想链中毒攻击 cs.IR

7 pages,3 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16367v1) [paper-pdf](http://arxiv.org/pdf/2505.16367v1)

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Yixing Fan

**Abstract**: Retrieval-augmented generation (RAG) systems can effectively mitigate the hallucination problem of large language models (LLMs),but they also possess inherent vulnerabilities. Identifying these weaknesses before the large-scale real-world deployment of RAG systems is of great importance, as it lays the foundation for building more secure and robust RAG systems in the future. Existing adversarial attack methods typically exploit knowledge base poisoning to probe the vulnerabilities of RAG systems, which can effectively deceive standard RAG models. However, with the rapid advancement of deep reasoning capabilities in modern LLMs, previous approaches that merely inject incorrect knowledge are inadequate when attacking RAG systems equipped with deep reasoning abilities. Inspired by the deep thinking capabilities of LLMs, this paper extracts reasoning process templates from R1-based RAG systems, uses these templates to wrap erroneous knowledge into adversarial documents, and injects them into the knowledge base to attack RAG systems. The key idea of our approach is that adversarial documents, by simulating the chain-of-thought patterns aligned with the model's training signals, may be misinterpreted by the model as authentic historical reasoning processes, thus increasing their likelihood of being referenced. Experiments conducted on the MS MARCO passage ranking dataset demonstrate the effectiveness of our proposed method.

摘要: 检索增强生成（RAG）系统可以有效地缓解大型语言模型（LLM）的幻觉问题，但它们也具有固有的漏洞。在RAG系统大规模现实部署之前识别这些弱点非常重要，因为它为未来构建更安全、更强大的RAG系统奠定了基础。现有的对抗攻击方法通常利用知识库中毒来探测RAG系统的漏洞，这可以有效地欺骗标准RAG模型。然而，随着现代LLM深度推理能力的迅速进步，以前仅仅注入错误知识的方法在攻击配备深度推理能力的RAG系统时是不够的。受LLM深度思维能力的启发，本文从基于R1的RAG系统中提取推理过程模板，使用这些模板将错误知识包装到对抗文档中，并将其注入知识库中以攻击RAG系统。我们方法的关键思想是，通过模拟与模型训练信号一致的思维链模式，对抗性文档可能会被模型误解为真实的历史推理过程，从而增加它们被引用的可能性。在MS MARCO通过排名数据集上进行的实验证明了我们提出的方法的有效性。



## **40. PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation**

PoisonArena：揭露检索增强一代中的竞争中毒攻击 cs.IR

29 pages

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.12574v3) [paper-pdf](http://arxiv.org/pdf/2505.12574v3)

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: https://github.com/yxf203/PoisonArena.

摘要: 检索增强生成（RAG）系统，广泛用于改善大型语言模型（LLM）的事实基础，越来越容易受到中毒攻击，其中对手将操纵的内容注入检索器的语料库。虽然以前的研究主要集中在单个攻击者的设置，但现实世界的场景往往涉及多个相互竞争的攻击者，这些攻击者的目标相互冲突。在这项工作中，我们介绍PoisonArena，第一个基准系统地研究和评估竞争中毒攻击在RAG。我们形式化的多攻击者威胁模型，攻击者争夺控制答案相同的查询使用互斥的错误信息。PoisonArena利用Bradley-Terry模型来量化每种方法在此类对抗环境中的竞争有效性。通过对Natural Questions和MS MARCO数据集的广泛实验，我们证明了许多孤立成功的攻击策略在竞争压力下失败。我们的研究结果强调了攻击成功率（SVR）和F1评分等传统评估指标的局限性，并强调了竞争性评估来评估现实世界攻击稳健性的必要性。PoisonArena提供了一个标准化的框架，可以在更现实的多对手条件下基准和开发未来的攻击和防御策略。项目页面：https://github.com/yxf203/PoisonArena。



## **41. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

PandaGuard：针对越狱攻击的LLM安全性系统评估 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.13862v2) [paper-pdf](http://arxiv.org/pdf/2505.13862v2)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.

摘要: 大型语言模型（LLM）已经取得了卓越的能力，但仍然容易受到被称为越狱的对抗性提示的影响，这可能会绕过安全对齐并引发有害的输出。尽管LLM安全研究的努力越来越多，但现有的评估往往是分散的，集中在孤立的攻击或防御技术上，缺乏系统的，可重复的分析。在这项工作中，我们引入了PandaGuard，一个统一的模块化框架，将LLM越狱安全建模为一个由攻击者，防御者和法官组成的多代理系统。我们的框架实现了19种攻击方法和12种防御机制，以及多种判断策略，所有这些都在一个灵活的插件架构中，支持多种LLM接口，多种交互模式和配置驱动的实验，从而增强了可重复性和实际部署。基于这个框架，我们开发了PandaBench，这是一个全面的基准，可评估49个LLM和各种判断方法之间的相互作用，需要超过30亿个代币来执行。我们的广泛评估揭示了对模型漏洞、国防成本-性能权衡和判断一致性的关键见解。我们发现，没有一种防御在所有维度上都是最佳的，而且判断分歧会在安全评估中引入非平凡的方差。我们发布代码、配置和评估结果，以支持LLM安全性方面的透明和可重复研究。



## **42. Robustifying Vision-Language Models via Dynamic Token Reweighting**

通过动态令牌重新加权来增强视觉语言模型 cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17132v1) [paper-pdf](http://arxiv.org/pdf/2505.17132v1)

**Authors**: Tanqiu Jiang, Jiacheng Liang, Rongyi Zhu, Jiawei Zhou, Fenglong Ma, Ting Wang

**Abstract**: Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. The code for replicating DTR is available: https://anonymous.4open.science/r/DTR-2755 (warning: this paper contains potentially harmful content generated by VLMs.)

摘要: 大型视觉语言模型（VLM）极易受到越狱攻击，这些攻击利用视觉与文本交互来绕过安全护栏。在本文中，我们提出了DTR，这是一种新型的推理时防御，通过优化模型的key-Value（KV）缓存来减轻多模式越狱攻击。我们不是依赖精心策划的安全特定数据或昂贵的图像到文本转换，而是引入了视觉模式引发的安全相关分布转变的新公式。该公式使DTR能够动态调整视觉令牌权重，最大限度地减少对抗视觉输入的影响，同时保留模型的一般能力和推理效率。对各种VLM和攻击基准的广泛评估表明，\sys在攻击稳健性和良性任务性能方面都优于现有防御，标志着在多模式基础模型中首次成功应用KV缓存优化来增强安全性。复制DTR的代码可获取：https://anonymous.4open.science/r/DTR-2755（警告：本文包含VLM生成的潜在有害内容。）



## **43. Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering**

保持安全！针对问题解答中的间接攻击，对大型语言模型上下文中的安全策略保留进行基准测试 cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15805v1) [paper-pdf](http://arxiv.org/pdf/2505.15805v1)

**Authors**: Hwan Chang, Yumin Kim, Yonghyun Jun, Hwanhee Lee

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains such as enterprise and government, ensuring that they adhere to user-defined security policies within context is critical-especially with respect to information non-disclosure. While prior LLM studies have focused on general safety and socially sensitive data, large-scale benchmarks for contextual security preservation against attacks remain lacking. To address this, we introduce a novel large-scale benchmark dataset, CoPriva, evaluating LLM adherence to contextual non-disclosure policies in question answering. Derived from realistic contexts, our dataset includes explicit policies and queries designed as direct and challenging indirect attacks seeking prohibited information. We evaluate 10 LLMs on our benchmark and reveal a significant vulnerability: many models violate user-defined policies and leak sensitive information. This failure is particularly severe against indirect attacks, highlighting a critical gap in current LLM safety alignment for sensitive applications. Our analysis reveals that while models can often identify the correct answer to a query, they struggle to incorporate policy constraints during generation. In contrast, they exhibit a partial ability to revise outputs when explicitly prompted. Our findings underscore the urgent need for more robust methods to guarantee contextual security.

摘要: 随着大型语言模型（LLM）越来越多地部署在企业和政府等敏感领域，确保它们在上下文中遵守用户定义的安全策略至关重要，尤其是在信息不披露方面。虽然之前的LLM研究重点关注一般安全和社会敏感数据，但仍然缺乏针对攻击的上下文安全保护的大规模基准。为了解决这个问题，我们引入了一个新颖的大规模基准数据集CoPriva，以评估LLM在问答中对上下文保密政策的遵守情况。我们的数据集源自现实背景，包括明确的政策和查询，旨在作为寻求违禁信息的直接和具有挑战性的间接攻击。我们在我们的基准上评估了10个LLM，并揭示了一个重大漏洞：许多模型违反了用户定义的策略并泄露了敏感信息。对于间接攻击，这种故障尤其严重，凸显了当前针对敏感应用的LLM安全调整中的关键差距。我们的分析表明，虽然模型通常可以识别查询的正确答案，但它们很难在生成过程中纳入政策约束。相比之下，它们在明确提示时表现出修改输出的部分能力。我们的研究结果强调迫切需要更强大的方法来保证上下文安全。



## **44. Reverse Engineering Human Preferences with Reinforcement Learning**

利用强化学习反向工程人类偏好 cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15795v1) [paper-pdf](http://arxiv.org/pdf/2505.15795v1)

**Authors**: Lisa Alazraki, Tan Yi-Chern, Jon Ander Campos, Maximilian Mozes, Marek Rei, Max Bartolo

**Abstract**: The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.

摘要: 大型语言模型（LLM）的能力通常由其他经过训练以预测人类偏好的LLM进行评估。这个框架-被称为LLM作为法官-具有高度可扩展性和相对较低的成本。然而，它也容易受到恶意利用，因为LLM响应可以被调整以过度适应法官的偏好。以前的工作表明，候选人LLM生成的答案可以事后编辑，以最大限度地提高法官LLM分配给他们的分数。在这项研究中，我们采用了一种不同的方法，并使用judge-LLM提供的信号作为奖励，以对抗性地调整模型，这些模型生成旨在提高下游性能的文本前置码。我们发现，使用这些模型流水线化的冻结LLM比现有框架获得更高的LLM评估分数。至关重要的是，与直接干预模型响应的其他框架不同，我们的方法几乎无法检测。我们还证明，当候选LLM和判断LLM被训练期间未使用的模型替换时，调整后的前同步码生成器的有效性会转移。这些发现提出了更可靠的法学硕士作为一个法官的评价设置的设计的重要问题。他们还证明，人类的偏好可以有效地进行逆向工程，通过流水线LLM来优化上游的优化，这种方法可以在对抗性攻击之外的各种任务和领域中找到未来的应用。



## **45. Scalable Defense against In-the-wild Jailbreaking Attacks with Safety Context Retrieval**

通过安全上下文检索针对野外越狱攻击的可扩展防御 cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15753v1) [paper-pdf](http://arxiv.org/pdf/2505.15753v1)

**Authors**: Taiye Chen, Zeming Wei, Ang Li, Yisen Wang

**Abstract**: Large Language Models (LLMs) are known to be vulnerable to jailbreaking attacks, wherein adversaries exploit carefully engineered prompts to induce harmful or unethical responses. Such threats have raised critical concerns about the safety and reliability of LLMs in real-world deployment. While existing defense mechanisms partially mitigate such risks, subsequent advancements in adversarial techniques have enabled novel jailbreaking methods to circumvent these protections, exposing the limitations of static defense frameworks. In this work, we explore defending against evolving jailbreaking threats through the lens of context retrieval. First, we conduct a preliminary study demonstrating that even a minimal set of safety-aligned examples against a particular jailbreak can significantly enhance robustness against this attack pattern. Building on this insight, we further leverage the retrieval-augmented generation (RAG) techniques and propose Safety Context Retrieval (SCR), a scalable and robust safeguarding paradigm for LLMs against jailbreaking. Our comprehensive experiments demonstrate how SCR achieves superior defensive performance against both established and emerging jailbreaking tactics, contributing a new paradigm to LLM safety. Our code will be available upon publication.

摘要: 众所周知，大型语言模型（LLM）很容易受到越狱攻击，其中对手利用精心设计的提示来引发有害或不道德的反应。此类威胁引发了人们对LLM在现实世界部署中的安全性和可靠性的严重担忧。虽然现有的防御机制部分减轻了此类风险，但对抗技术的后续进步使新型越狱方法能够规避这些保护，暴露了静态防御框架的局限性。在这项工作中，我们探索通过上下文检索的视角抵御不断变化的越狱威胁。首先，我们进行了一项初步研究，证明即使是针对特定越狱的最少一组安全一致的示例也可以显着增强针对这种攻击模式的鲁棒性。在这一见解的基础上，我们进一步利用检索增强生成（RAG）技术并提出安全上下文检索（SR），这是一种针对LLM越狱的可扩展且强大的保护范式。我们全面的实验展示了可控硅如何在针对既定和新兴越狱策略的情况下实现卓越的防御性能，为LLM安全性贡献了新的范式。我们的代码将在发布后提供。



## **46. Shaping the Safety Boundaries: Understanding and Defending Against Jailbreaks in Large Language Models**

塑造安全边界：理解和防御大型语言模型中的越狱 cs.CL

17 pages, 9 figures

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2412.17034v2) [paper-pdf](http://arxiv.org/pdf/2412.17034v2)

**Authors**: Lang Gao, Jiahui Geng, Xiangliang Zhang, Preslav Nakov, Xiuying Chen

**Abstract**: Jailbreaking in Large Language Models (LLMs) is a major security concern as it can deceive LLMs to generate harmful text. Yet, there is still insufficient understanding of how jailbreaking works, which makes it hard to develop effective defense strategies. We aim to shed more light into this issue: we conduct a detailed large-scale analysis of seven different jailbreak methods and find that these disagreements stem from insufficient observation samples. In particular, we introduce \textit{safety boundary}, and we find that jailbreaks shift harmful activations outside that safety boundary, where LLMs are less sensitive to harmful information. We also find that the low and the middle layers are critical in such shifts, while deeper layers have less impact. Leveraging on these insights, we propose a novel defense called \textbf{Activation Boundary Defense} (ABD), which adaptively constrains the activations within the safety boundary. We further use Bayesian optimization to selectively apply the defense method to the low and the middle layers. Our experiments on several benchmarks show that ABD achieves an average DSR of over 98\% against various forms of jailbreak attacks, with less than 2\% impact on the model's general capabilities.

摘要: 大型语言模型（LLM）中的越狱是一个主要的安全问题，因为它可能会欺骗LLM生成有害文本。然而，人们对越狱的运作方式仍然缺乏足够的了解，这使得制定有效的防御策略变得困难。我们的目标是更多地了解这个问题：我们对七种不同的越狱方法进行了详细的大规模分析，发现这些分歧源于观察样本不足。特别是，我们引入了\textit{safety boundary}，我们发现越狱将有害激活转移到安全边界之外，而LLM对有害信息不太敏感。我们还发现，低层和中层在此类转变中至关重要，而较深层的影响较小。利用这些见解，我们提出了一种名为\textBF{Activation Boundary Defense}（ABD）的新型防御，它自适应地将激活限制在安全边界内。我们进一步使用Bayesian优化来选择性地将防御方法应用于低层和中层。我们在多个基准测试上的实验表明，ABD针对各种形式的越狱攻击，平均DSR超过98%，对模型的一般能力影响不到2%。



## **47. Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses**

压力下的一致：评估LLM防御时知情对手的理由 cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15738v1) [paper-pdf](http://arxiv.org/pdf/2505.15738v1)

**Authors**: Xiaoxue Yang, Bozhidar Stevanoski, Matthieu Meeus, Yves-Alexandre de Montjoye

**Abstract**: Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs. Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs.

摘要: 大型语言模型（LLM）被快速部署在从聊天机器人到代理系统的实际应用中。对齐是用于防御诸如即时注入和越狱等攻击的主要方法之一。最近的防御报告甚至对贪婪坐标梯度（GCG）的攻击成功率（ASR）接近于零，GCG是一种白盒攻击，生成对抗性后缀以诱导攻击者期望的输出。然而，这种在离散令牌上的搜索空间非常大，使得找到成功攻击的任务变得困难。例如，GCG已被证明收敛到局部极小值，使其对初始化选择敏感。在本文中，我们使用一个更明智的威胁模型来评估这些防御系统的面向未来的鲁棒性：可以访问有关对齐过程的一些信息的攻击者。具体来说，我们提出了一种知情白盒攻击，利用中间模型检查点来初始化GCG，每个检查点都充当下一个检查点的垫脚石。我们证明这种方法在最先进的（SOTA）防御和模型中非常有效。我们进一步展示了我们的知情初始化，以优于其他初始化方法，并展示了一种基于梯度的检查点选择策略，以极大地提高攻击性能和效率。重要的是，我们还展示了成功找到通用对抗后缀的方法--在不同输入中有效的单个后缀。我们的结果表明，与之前的观点相反，针对基于SOTA匹配的防御，确实存在有效的对抗性后缀，当对手利用对齐知识时，这些后缀可以通过现有的攻击方法找到，甚至存在通用后缀。总而言之，我们的结果凸显了当前基于环境的方法的脆弱性，以及在测试LLM的安全性时需要考虑更强的威胁模型。



## **48. SQL Injection Jailbreak: A Structural Disaster of Large Language Models**

SQL注入越狱：大型语言模型的结构灾难 cs.CR

Accepted by findings of ACL 2025

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2411.01565v6) [paper-pdf](http://arxiv.org/pdf/2411.01565v6)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: Large Language Models (LLMs) are susceptible to jailbreak attacks that can induce them to generate harmful content. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak methods and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. For open-source models, SIJ achieves near 100% attack success rates on five well-known LLMs on the AdvBench and HEx-PHI, while incurring lower time costs compared to previous methods. For closed-source models, SIJ achieves an average attack success rate over 85% across five models in the GPT and Doubao series. Additionally, SIJ exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple adaptive defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at https://github.com/weiyezhimeng/SQL-Injection-Jailbreak.

摘要: 大型语言模型（LLM）容易受到越狱攻击，从而导致它们生成有害内容。之前的越狱方法主要利用LLM的内部属性或功能，例如基于优化的越狱方法和利用模型上下文学习能力的方法。本文中，我们介绍了一种新颖的越狱方法--SQL注入越狱（SIJ），它针对的是LLM的外部属性，具体来说是LLM构建输入提示的方式。通过将越狱信息注入用户提示中，SIJ成功诱导模型输出有害内容。对于开源模型，SIJ在AdvBench和HEx-PHI上的五个知名LLM上实现了接近100%的攻击成功率，同时与之前的方法相比，时间成本更低。对于闭源型号，SIJ在GPT和抖音系列的五种型号中的平均攻击成功率超过85%。此外，SIJ暴露了LLM中的一个新漏洞，迫切需要缓解。为了解决这个问题，我们提出了一种名为Self-Reminder-Key的简单自适应防御方法来对抗SIJ，并通过实验结果证明其有效性。我们的代码可在https://github.com/weiyezhimeng/SQL-Injection-Jailbreak上获取。



## **49. Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!**

在开源LLM上进行微调时要小心：您的微调数据可能会被秘密窃取！ cs.CL

19 pages

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15656v1) [paper-pdf](http://arxiv.org/pdf/2505.15656v1)

**Authors**: Zhexin Zhang, Yuhao Sun, Junxiao Yang, Shiyao Cui, Hongning Wang, Minlie Huang

**Abstract**: Fine-tuning on open-source Large Language Models (LLMs) with proprietary data is now a standard practice for downstream developers to obtain task-specific LLMs. Surprisingly, we reveal a new and concerning risk along with the practice: the creator of the open-source LLMs can later extract the private downstream fine-tuning data through simple backdoor training, only requiring black-box access to the fine-tuned downstream model. Our comprehensive experiments, across 4 popularly used open-source models with 3B to 32B parameters and 2 downstream datasets, suggest that the extraction performance can be strikingly high: in practical settings, as much as 76.3% downstream fine-tuning data (queries) out of a total 5,000 samples can be perfectly extracted, and the success rate can increase to 94.9% in more ideal settings. We also explore a detection-based defense strategy but find it can be bypassed with improved attack. Overall, we highlight the emergency of this newly identified data breaching risk in fine-tuning, and we hope that more follow-up research could push the progress of addressing this concerning risk. The code and data used in our experiments are released at https://github.com/thu-coai/Backdoor-Data-Extraction.

摘要: 对具有专有数据的开源大型语言模型（LLM）进行微调现在已成为下游开发人员获取特定任务LLM的标准实践。令人惊讶的是，我们在实践中揭示了一个新的且令人担忧的风险：开源LLM的创建者稍后可以通过简单的后门训练提取私有下游微调数据，只需要黑匣子访问微调下游模型。我们对4个常用的3B至32 B参数开源模型和2个下游数据集进行了全面的实验，表明提取性能可以非常高：在实际环境中，总共5，000个样本中，多达76.3%的下游微调数据（查询）可以被完美提取，在更理想的环境中，成功率可以提高到94.9%。我们还探索了基于检测的防御策略，但发现可以通过改进的攻击来绕过它。总体而言，我们强调了这种新发现的数据泄露风险在微调中的紧迫性，我们希望更多的后续研究能够推动解决这一相关风险的进展。我们实验中使用的代码和数据发布在https://github.com/thu-coai/Backdoor-Data-Extraction上。



## **50. SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings**

SEA：通过合成嵌入实现多模式大型语言模型的低资源安全性对齐 cs.CL

Accepted in ACL 2025 Main Track

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2502.12562v2) [paper-pdf](http://arxiv.org/pdf/2502.12562v2)

**Authors**: Weikai Lu, Hao Peng, Huiping Zhuang, Cen Chen, Ziqian Zeng

**Abstract**: Multimodal Large Language Models (MLLMs) have serious security vulnerabilities.While safety alignment using multimodal datasets consisting of text and data of additional modalities can effectively enhance MLLM's security, it is costly to construct these datasets. Existing low-resource security alignment methods, including textual alignment, have been found to struggle with the security risks posed by additional modalities. To address this, we propose Synthetic Embedding augmented safety Alignment (SEA), which optimizes embeddings of additional modality through gradient updates to expand textual datasets. This enables multimodal safety alignment training even when only textual data is available. Extensive experiments on image, video, and audio-based MLLMs demonstrate that SEA can synthesize a high-quality embedding on a single RTX3090 GPU within 24 seconds. SEA significantly improves the security of MLLMs when faced with threats from additional modalities. To assess the security risks introduced by video and audio, we also introduced a new benchmark called VA-SafetyBench. High attack success rates across multiple MLLMs validate its challenge. Our code and data will be available at https://github.com/ZeroNLP/SEA.

摘要: 多模式大型语言模型（MLLM）存在严重的安全漏洞。虽然使用由文本和其他模式数据组成的多模式数据集进行安全对齐可以有效增强MLLM的安全性，但构建这些数据集的成本很高。现有的低资源安全对齐方法（包括文本对齐）被发现难以应对额外模式带来的安全风险。为了解决这个问题，我们提出了合成嵌入增强安全对齐（SEA），它通过梯度更新来优化额外模式的嵌入以扩展文本数据集。即使只有文本数据可用，这也可以实现多模式安全对齐训练。基于图像、视频和音频的MLLM的广泛实验表明，SEA可以在24秒内在单个RTX 3090图形处理器上合成高质量嵌入。SEA在面临来自其他模式的威胁时显着提高了MLLM的安全性。为了评估视频和音频带来的安全风险，我们还引入了名为VA-SafetyBench的新基准。多个MLLM的高攻击成功率证实了其挑战。我们的代码和数据可在https://github.com/ZeroNLP/SEA上获取。




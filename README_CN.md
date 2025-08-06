# Latest Adversarial Attack Papers
**update at 2025-08-06 18:23:38**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. LeakyCLIP: Extracting Training Data from CLIP**

LeakyCLIP：从CLIP中提取训练数据 cs.CR

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.00756v2) [paper-pdf](http://arxiv.org/pdf/2508.00756v2)

**Authors**: Yunhao Chen, Shujie Wang, Xin Wang, Xingjun Ma

**Abstract**: Understanding the memorization and privacy leakage risks in Contrastive Language--Image Pretraining (CLIP) is critical for ensuring the security of multimodal models. Recent studies have demonstrated the feasibility of extracting sensitive training examples from diffusion models, with conditional diffusion models exhibiting a stronger tendency to memorize and leak information. In this work, we investigate data memorization and extraction risks in CLIP through the lens of CLIP inversion, a process that aims to reconstruct training images from text prompts. To this end, we introduce \textbf{LeakyCLIP}, a novel attack framework designed to achieve high-quality, semantically accurate image reconstruction from CLIP embeddings. We identify three key challenges in CLIP inversion: 1) non-robust features, 2) limited visual semantics in text embeddings, and 3) low reconstruction fidelity. To address these challenges, LeakyCLIP employs 1) adversarial fine-tuning to enhance optimization smoothness, 2) linear transformation-based embedding alignment, and 3) Stable Diffusion-based refinement to improve fidelity. Empirical results demonstrate the superiority of LeakyCLIP, achieving over 358% improvement in Structural Similarity Index Measure (SSIM) for ViT-B-16 compared to baseline methods on LAION-2B subset. Furthermore, we uncover a pervasive leakage risk, showing that training data membership can even be successfully inferred from the metrics of low-fidelity reconstructions. Our work introduces a practical method for CLIP inversion while offering novel insights into the nature and scope of privacy risks in multimodal models.

摘要: 了解对比语言-图像预训练（CLIP）中的记忆和隐私泄露风险对于确保多模式模型的安全性至关重要。最近的研究证明了从扩散模型中提取敏感训练示例的可行性，条件扩散模型表现出更强的记忆和泄露信息的倾向。在这项工作中，我们通过CLIP倒置的镜头调查了CLIP中的数据记忆和提取风险，这是一个旨在根据文本提示重建训练图像的过程。为此，我们引入了\textBF{LeakyCLIP}，这是一种新型攻击框架，旨在从CLIP嵌入中实现高质量、语义准确的图像重建。我们确定了CLIP倒置中的三个关键挑战：1）非鲁棒特征，2）文本嵌入中的视觉语义有限，以及3）重建保真度低。为了解决这些挑战，LeakyCLIP采用1）对抗性微调以增强优化平滑度，2）基于线性变换的嵌入对齐，以及3）基于稳定扩散的细化以提高保真度。经验结果证明了LeakyCLIP的优越性，与LAION-2B子集的基线方法相比，ViT-B-16的结构相似性指数测量（SSIM）提高了358%以上。此外，我们还发现了普遍存在的泄露风险，表明训练数据成员关系甚至可以从低保真重建的指标中成功推断出来。我们的工作介绍了一种实用的方法CLIP反演，同时提供了新的见解的性质和范围的隐私风险的多模态模型。



## **2. Set-Based Training for Neural Network Verification**

神经网络验证的基于集的训练 cs.LG

published at Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2401.14961v4) [paper-pdf](http://arxiv.org/pdf/2401.14961v4)

**Authors**: Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can significantly affect the outputs of a neural network. Therefore, to ensure safety of neural networks in safety-critical environments, the robustness of a neural network must be formally verified against input perturbations, e.g., from noisy sensors. To improve the robustness of neural networks and thus simplify the formal verification, we present a novel set-based training procedure in which we compute the set of possible outputs given the set of possible inputs and compute for the first time a gradient set, i.e., each possible output has a different gradient. Therefore, we can directly reduce the size of the output enclosure by choosing gradients toward its center. Small output enclosures increase the robustness of a neural network and, at the same time, simplify its formal verification. The latter benefit is due to the fact that a larger size of propagated sets increases the conservatism of most verification methods. Our extensive evaluation demonstrates that set-based training produces robust neural networks with competitive performance, which can be verified using fast (polynomial-time) verification algorithms due to the reduced output set.

摘要: 神经网络容易受到对抗攻击，即小的输入扰动会显着影响神经网络的输出。因此，为了确保神经网络在安全关键环境中的安全性，必须针对输入扰动正式验证神经网络的鲁棒性，例如，来自有噪音的传感器。为了提高神经网络的鲁棒性并从而简化形式验证，我们提出了一种新颖的基于集合的训练过程，其中我们在给定可能输入集合的情况下计算可能输出集合，并首次计算梯度集合，即，每个可能的输出都有不同的梯度。因此，我们可以通过选择向输出外壳中心的梯度来直接缩小输出外壳的尺寸。小输出外壳增强了神经网络的稳健性，同时简化了其形式验证。后一个好处是由于传播集的更大大小会增加大多数验证方法的保守性。我们的广泛评估表明，基于集合的训练可以产生具有竞争力性能的鲁棒神经网络，由于输出集合减少，可以使用快速（多项时间）验证算法来验证该网络。



## **3. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

IDEATOR：使用自己越狱和基准大型视觉语言模型 cs.CV

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2411.00827v4) [paper-pdf](http://arxiv.org/pdf/2411.00827v4)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLJailbreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.

摘要: 随着大型视觉语言模型（VLM）的日益突出，确保其安全部署变得至关重要。最近的研究探索了VLM针对越狱攻击的鲁棒性--利用模型漏洞来引发有害输出的技术。然而，多样化多模式数据的可用性有限，限制了当前的方法严重依赖于从有害文本数据集派生的对抗性或手动制作的图像，而这些图像通常缺乏跨不同背景的有效性和多样性。本文中，我们提出了IDEATOR，这是一种新型越狱方法，可以自主生成用于黑匣子越狱攻击的恶意图像-文本对。IDEATOR基于这样的见解：VLM本身可以充当强大的红队模型，用于生成多模式越狱提示。具体来说，IDEATOR利用VLM创建有针对性的越狱文本，并将其与由最先进的扩散模型生成的越狱图像配对。大量实验证明了IDEATOR的高效率和可移植性，在越狱MiniGPT-4中平均只需5.34次查询即可实现94%的攻击成功率（ASB），转移到LLaVA、INSTBLIP和Chameleon时，攻击成功率分别为82%、88%和75%。基于IDEATOR强大的可移植性和自动化流程，我们推出了VLJailbreakBench，这是一个由3，654个多模式越狱样本组成的安全基准。我们对最近发布的11个VLM的基准结果揭示了安全一致方面的显着差距。例如，我们的挑战集在GPT-4 o上实现了46.31%的ASB，在Claude-3.5-十四行诗上实现了19.65%的ASB，这凸显了迫切需要更强的防御。



## **4. Smart Car Privacy: Survey of Attacks and Privacy Issues**

智能汽车隐私：攻击和隐私问题调查 cs.CR

13 pages, 16 figures

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.03413v1) [paper-pdf](http://arxiv.org/pdf/2508.03413v1)

**Authors**: Akshay Madhav Deshmukh

**Abstract**: Automobiles are becoming increasingly important in our day to day life. Modern automobiles are highly computerized and hence potentially vulnerable to attack. Providing many wireless connectivity for vehicles enables a bridge between vehicles and their external environments. Such a connected vehicle solution is expected to be the next frontier for automotive revolution and the key to the evolution to next generation intelligent transportation systems. Vehicular Ad hoc Networks (VANETs) are emerging mobile ad hoc network technologies incorporating mobile routing protocols for inter-vehicle data communications to support intelligent transportation systems. Thus security and privacy are the major concerns in VANETs due to the mobility of the vehicles. Thus designing security mechanisms to remove adversaries from the network remarkably important in VANETs.   This paper provides an overview of various vehicular network architectures. The evolution of security in modern vehicles. Various security and privacy attacks in VANETs with their defending mechanisms with examples and classify these mechanisms. It also provides an overview of various privacy implication that a vehicular network possess.

摘要: 汽车在我们的日常生活中变得越来越重要。现代汽车高度计算机化，因此可能容易受到攻击。为车辆提供许多无线连接可以在车辆与其外部环境之间架起桥梁。这种互联汽车解决方案有望成为汽车革命的下一个前沿，也是下一代智能交通系统演变的关键。车载自组织网络（VANSYS）是新兴的移动自组织网络技术，结合了用于车辆间数据通信的移动路由协议，以支持智能交通系统。因此，由于车辆的移动性，安全和隐私是VANSYS的主要担忧。因此，设计安全机制以将对手从网络中删除在VANSYS中显得非常重要。   本文概述了各种车载网络架构。现代车辆安全性的演变。VANSYS中的各种安全和隐私攻击及其防御机制，并举例并对这些机制进行分类。它还概述了车辆网络具有的各种隐私含义。



## **5. When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs**

当好的声音变得敌对时：用良性输入越狱的音频模型 cs.SD

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.03365v1) [paper-pdf](http://arxiv.org/pdf/2508.03365v1)

**Authors**: Bodam Kim, Hiskias Dingeto, Taeyoun Kwon, Dasol Choi, DongGeon Lee, Haon Park, JaeHoon Lee, Jongho Shin

**Abstract**: As large language models become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that can manipulate state-of-the-art audio language models to generate harmful content. Our method uses imperceptible perturbations in audio inputs that remain benign to human listeners. The first stage uses a novel reward-based optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to guide the target model to circumvent its own safety protocols and generate harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use Projected Gradient Descent (PGD) to optimize subtle perturbations that are embedded into benign audio carriers, such as weather queries or greeting messages. Validated under the rigorous StrongREJECT, LlamaGuard, as well as Human Evaluation safety evaluation framework, our experiments demonstrate a success rate exceeding 86% across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating AI behavior.

摘要: 随着大型语言模型越来越融入日常生活，音频已成为人机交互的关键界面。然而，这种便利性也引入了新的漏洞，使音频成为对手的潜在攻击面。我们的研究引入了WhisperInib，这是一个两阶段对抗性音频攻击框架，可以操纵最先进的音频语言模型来生成有害内容。我们的方法在音频输入中使用不可感知的扰动，这些扰动对人类听众保持良性。第一阶段使用一种新颖的基于奖励的优化方法--具有投影梯度下降的强化学习（RL-PVD），来指导目标模型规避其自己的安全协议并生成有害的原生响应。然后，这种原生有害响应作为第二阶段有效负载注入的目标，在该阶段，我们使用投影梯度下降（PVD）来优化嵌入良性音频载体中的微妙扰动，例如天气查询或问候消息。我们的实验经过严格的StrongRESEARCH、LlamaGuard以及Human Evision安全评估框架的验证，证明Qwen 2.5-Omni-3B、Qwen 2.5-Omni-7 B和Phi-4-Multimodal的成功率超过86%。我们的工作展示了一类新的实用、音频原生威胁，超越了理论利用，揭示了一种可行且隐蔽的操纵人工智能行为的方法。



## **6. LADSG: Label-Anonymized Distillation and Similar Gradient Substitution for Label Privacy in Vertical Federated Learning**

LADSG：垂直联邦学习中标签模拟蒸馏和标签隐私的类似梯度替代 cs.CR

Under review

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2506.06742v2) [paper-pdf](http://arxiv.org/pdf/2506.06742v2)

**Authors**: Zeyu Yan, Yifei Yao, Xuanbing Wen, Shixiong Zhang, Juli Zhang, Kai Fan

**Abstract**: Vertical Federated Learning (VFL) has emerged as a promising paradigm for collaborative model training across distributed feature spaces, which enables privacy-preserving learning without sharing raw data. However, recent studies have confirmed the feasibility of label inference attacks by internal adversaries. By strategically exploiting gradient vectors and semantic embeddings, attackers-through passive, active, or direct attacks-can accurately reconstruct private labels, leading to catastrophic data leakage. Existing defenses, which typically address isolated leakage vectors or are designed for specific types of attacks, remain vulnerable to emerging hybrid attacks that exploit multiple pathways simultaneously. To bridge this gap, we propose Label-Anonymized Defense with Substitution Gradient (LADSG), a unified and lightweight defense framework for VFL. LADSG first anonymizes true labels via soft distillation to reduce semantic exposure, then generates semantically-aligned substitute gradients to disrupt gradient-based leakage, and finally filters anomalous updates through gradient norm detection. It is scalable and compatible with standard VFL pipelines. Extensive experiments on six real-world datasets show that LADSG reduces the success rates of all three types of label inference attacks by 30-60% with minimal computational overhead, demonstrating its practical effectiveness.

摘要: 垂直联邦学习（VFL）已成为跨分布式特征空间协作模型训练的一种有前途的范式，它可以在无需共享原始数据的情况下实现隐私保护学习。然而，最近的研究证实了内部对手进行标签推断攻击的可行性。通过战略性地利用梯度载体和语义嵌入，攻击者通过被动、主动或直接攻击可以准确地重建私有标签，从而导致灾难性的数据泄露。现有的防御系统通常针对孤立的泄漏载体或专为特定类型的攻击而设计，但仍然容易受到同时利用多个途径的新兴混合攻击的影响。为了弥合这一差距，我们提出了标签化替代梯度防御（LADSG），一个统一的，轻量级的VFL防御框架。LADSG首先通过软蒸馏匿名化真实标签以减少语义暴露，然后生成语义对齐的替代梯度以破坏基于梯度的泄漏，最后通过梯度范数检测过滤异常更新。它是可扩展的，并与标准VFL管道兼容。在六个真实数据集上的大量实验表明，LADSG以最小的计算开销将所有三种类型的标签推理攻击的成功率降低了30-60%，证明了其实际有效性。



## **7. BlockA2A: Towards Secure and Verifiable Agent-to-Agent Interoperability**

BlockA2A：迈向安全且可验证的代理对代理互操作性 cs.CR

43 pages

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.01332v2) [paper-pdf](http://arxiv.org/pdf/2508.01332v2)

**Authors**: Zhenhua Zou, Zhuotao Liu, Lepeng Zhao, Qiuyang Zhan

**Abstract**: The rapid adoption of agentic AI, powered by large language models (LLMs), is transforming enterprise ecosystems with autonomous agents that execute complex workflows. Yet we observe several key security vulnerabilities in LLM-driven multi-agent systems (MASes): fragmented identity frameworks, insecure communication channels, and inadequate defenses against Byzantine agents or adversarial prompts. In this paper, we present the first systematic analysis of these emerging multi-agent risks and explain why the legacy security strategies cannot effectively address these risks. Afterwards, we propose BlockA2A, the first unified multi-agent trust framework that enables secure and verifiable and agent-to-agent interoperability. At a high level, BlockA2A adopts decentralized identifiers (DIDs) to enable fine-grained cross-domain agent authentication, blockchain-anchored ledgers to enable immutable auditability, and smart contracts to dynamically enforce context-aware access control policies. BlockA2A eliminates centralized trust bottlenecks, ensures message authenticity and execution integrity, and guarantees accountability across agent interactions. Furthermore, we propose a Defense Orchestration Engine (DOE) that actively neutralizes attacks through real-time mechanisms, including Byzantine agent flagging, reactive execution halting, and instant permission revocation. Empirical evaluations demonstrate BlockA2A's effectiveness in neutralizing prompt-based, communication-based, behavioral and systemic MAS attacks. We formalize its integration into existing MAS and showcase a practical implementation for Google's A2A protocol. Experiments confirm that BlockA2A and DOE operate with sub-second overhead, enabling scalable deployment in production LLM-based MAS environments.

摘要: 由大型语言模型（LLM）支持的代理人工智能的快速采用正在通过执行复杂工作流程的自主代理改变企业生态系统。然而，我们在LLM驱动的多代理系统（MASes）中观察到了几个关键的安全漏洞：碎片化的身份框架、不安全的通信渠道以及对拜占庭代理或对抗提示的防御不足。在本文中，我们对这些新出现的多代理风险进行了首次系统分析，并解释了为什么传统安全策略无法有效应对这些风险。随后，我们提出了BlockA2A，这是第一个统一的多代理信任框架，可以实现安全、可验证以及代理与代理的互操作性。在高层面上，BlockA2A采用去中心化标识符（DID）来实现细粒度的跨域代理认证，采用区块链锚定分类帐来实现不可变的可互换性，并采用智能合同来动态执行上下文感知的访问控制策略。BlockA2A消除了集中式信任瓶颈，确保消息真实性和执行完整性，并保证跨代理交互的问责制。此外，我们还提出了一种国防规划引擎（DOE），它通过实时机制主动中和攻击，包括拜占庭代理标记、反应式执行停止和即时许可撤销。经验评估证明BlockA2A在中和基于预算、基于通信的行为和系统性MAS攻击方面的有效性。我们将其正式集成到现有MAS中，并展示了Google A2 A协议的实际实现。实验证实BlockA2A和DOE的运行成本为亚秒级，从而能够在基于LLM的生产MAS环境中进行可扩展部署。



## **8. ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models**

ConfGuard：大型语言模型简单有效的后门检测 cs.CR

Under review

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.01365v2) [paper-pdf](http://arxiv.org/pdf/2508.01365v2)

**Authors**: Zihan Wang, Rui Zhang, Hongwei Li, Wenshu Fan, Wenbo Jiang, Qingchuan Zhao, Guowen Xu

**Abstract**: Backdoor attacks pose a significant threat to Large Language Models (LLMs), where adversaries can embed hidden triggers to manipulate LLM's outputs. Most existing defense methods, primarily designed for classification tasks, are ineffective against the autoregressive nature and vast output space of LLMs, thereby suffering from poor performance and high latency. To address these limitations, we investigate the behavioral discrepancies between benign and backdoored LLMs in output space. We identify a critical phenomenon which we term sequence lock: a backdoored model generates the target sequence with abnormally high and consistent confidence compared to benign generation. Building on this insight, we propose ConfGuard, a lightweight and effective detection method that monitors a sliding window of token confidences to identify sequence lock. Extensive experiments demonstrate ConfGuard achieves a near 100\% true positive rate (TPR) and a negligible false positive rate (FPR) in the vast majority of cases. Crucially, the ConfGuard enables real-time detection almost without additional latency, making it a practical backdoor defense for real-world LLM deployments.

摘要: 后门攻击对大型语言模型（LLM）构成重大威胁，对手可以嵌入隐藏触发器来操纵LLM的输出。大多数现有的防御方法主要是为分类任务设计的，对LLM的自回归性质和巨大的输出空间无效，从而遭受性能差和延迟高的影响。为了解决这些限制，我们调查了输出空间中良性和后门LLM之间的行为差异。我们发现了一个关键现象，我们称之为序列锁：与良性生成相比，后门模型以异常高且一致的置信度生成目标序列。基于这一见解，我们提出了ConfGuard，这是一种轻量级且有效的检测方法，可以监控令牌置信度的滑动窗口以识别序列锁。大量实验表明，在绝大多数情况下，ConfGuard的真阳性率（TPA）接近100%，假阳性率（FPR）可忽略不计。至关重要的是，ConfGuard几乎无需额外延迟即可实现实时检测，使其成为现实世界LLM部署的实用后门防御。



## **9. ProARD: progressive adversarial robustness distillation: provide wide range of robust students**

ProARD：渐进式对抗稳健性蒸馏：提供广泛的稳健学生 cs.LG

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2506.07666v2) [paper-pdf](http://arxiv.org/pdf/2506.07666v2)

**Authors**: Seyedhamidreza Mousavi, Seyedali Mousavi, Masoud Daneshtalab

**Abstract**: Adversarial Robustness Distillation (ARD) has emerged as an effective method to enhance the robustness of lightweight deep neural networks against adversarial attacks. Current ARD approaches have leveraged a large robust teacher network to train one robust lightweight student. However, due to the diverse range of edge devices and resource constraints, current approaches require training a new student network from scratch to meet specific constraints, leading to substantial computational costs and increased CO2 emissions. This paper proposes Progressive Adversarial Robustness Distillation (ProARD), enabling the efficient one-time training of a dynamic network that supports a diverse range of accurate and robust student networks without requiring retraining. We first make a dynamic deep neural network based on dynamic layers by encompassing variations in width, depth, and expansion in each design stage to support a wide range of architectures. Then, we consider the student network with the largest size as the dynamic teacher network. ProARD trains this dynamic network using a weight-sharing mechanism to jointly optimize the dynamic teacher network and its internal student networks. However, due to the high computational cost of calculating exact gradients for all the students within the dynamic network, a sampling mechanism is required to select a subset of students. We show that random student sampling in each iteration fails to produce accurate and robust students.

摘要: 对抗鲁棒性蒸馏（ARD）已成为增强轻量级深度神经网络抵御对抗攻击鲁棒性的有效方法。当前的ARD方法利用了一个强大的教师网络来培训一个强大的轻量级学生。然而，由于边缘设备的多样性和资源限制，当前的方法需要从头开始训练新的学生网络以满足特定的限制，从而导致巨大的计算成本和二氧化碳排放量增加。本文提出了渐进对抗鲁棒蒸馏（ProARD），可以对动态网络进行高效的一次性训练，该网络支持各种准确且稳健的学生网络，而无需再培训。我们首先基于动态层构建动态深度神经网络，通过涵盖每个设计阶段的宽度、深度和扩展的变化，以支持广泛的架构。然后，我们将规模最大的学生网络视为动态教师网络。ProARD使用权重共享机制训练这个动态网络，以联合优化动态教师网络及其内部学生网络。然而，由于计算动态网络中所有学生的精确梯度的计算成本很高，因此需要采样机制来选择学生的子集。我们表明，每次迭代中的随机学生抽样无法产生准确和稳健的学生。



## **10. M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs**

M2S：LLM红色团队多回合到单回合越狱 cs.CL

Accepted to ACL 2025 (Main Track). Camera-ready version

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2503.04856v3) [paper-pdf](http://arxiv.org/pdf/2503.04856v3)

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim

**Abstract**: We introduce a novel framework for consolidating multi-turn adversarial ``jailbreak'' prompts into single-turn queries, significantly reducing the manual overhead required for adversarial testing of large language models (LLMs). While multi-turn human jailbreaks have been shown to yield high attack success rates, they demand considerable human effort and time. Our multi-turn-to-single-turn (M2S) methods -- Hyphenize, Numberize, and Pythonize -- systematically reformat multi-turn dialogues into structured single-turn prompts. Despite removing iterative back-and-forth interactions, these prompts preserve and often enhance adversarial potency: in extensive evaluations on the Multi-turn Human Jailbreak (MHJ) dataset, M2S methods achieve attack success rates from 70.6 percent to 95.9 percent across several state-of-the-art LLMs. Remarkably, the single-turn prompts outperform the original multi-turn attacks by as much as 17.5 percentage points while cutting token usage by more than half on average. Further analysis shows that embedding malicious requests in enumerated or code-like structures exploits ``contextual blindness'', bypassing both native guardrails and external input-output filters. By converting multi-turn conversations into concise single-turn prompts, the M2S framework provides a scalable tool for large-scale red teaming and reveals critical weaknesses in contemporary LLM defenses.

摘要: 我们引入了一种新颖的框架，用于将多轮对抗性“越狱”提示整合到单轮查询中，从而显着减少了大型语言模型（LLM）对抗性测试所需的手动负担。虽然多回合人类越狱已被证明具有很高的攻击成功率，但它们需要相当大的人力和时间。我们的多回合到单回合（M2 S）方法--连字符化、数字化和Python化--系统地将多回合对话重新格式化为结构化的单回合提示。尽管消除了迭代的来回相互作用，但这些提示仍然保留并经常增强对抗能力：在对多回合人类越狱（MTJ）数据集的广泛评估中，M2 S方法在几种最先进的LLM中实现了从70.6%到95.9%的攻击成功率。值得注意的是，单回合提示的性能比最初的多回合攻击高出17.5个百分点，同时平均将代币使用量减少一半以上。进一步的分析表明，将恶意请求嵌入到列举或类代码结构中利用了“上下文盲目性”，绕过了本地护栏和外部输入输出过滤器。通过将多回合对话转换为简洁的单回合提示，M2 S框架为大规模红色团队提供了可扩展的工具，并揭示了当代LLM防御中的关键弱点。



## **11. Towards Imperceptible JPEG Image Hiding: Multi-range Representations-driven Adversarial Stego Generation**

迈向不可感知的JPEG图像隐藏：多范围表示驱动的对抗性Stego生成 cs.CV

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2507.08343v2) [paper-pdf](http://arxiv.org/pdf/2507.08343v2)

**Authors**: Junxue Yang, Xin Liao, Weixuan Tang, Jianhua Yang, Zheng Qin

**Abstract**: Image hiding fully explores the hidden potential of deep learning-based models, aiming to conceal image-level messages within cover images and reveal them from stego images to achieve covert communication. Existing hiding schemes are easily detected by the naked eyes or steganalyzers due to the cover type confined to the spatial domain, single-range feature extraction and attacks, and insufficient loss constraints. To address these issues, we propose a multi-range representations-driven adversarial stego generation framework called MRAG for JPEG image hiding. This design stems from the fact that steganalyzers typically combine local-range and global-range information to better capture hidden traces. Specifically, MRAG integrates the local-range characteristic of the convolution and the global-range modeling of the transformer. Meanwhile, a features angle-norm disentanglement loss is designed to launch multi-range representations-driven feature-level adversarial attacks. It computes the adversarial loss between covers and stegos based on the surrogate steganalyzer's classified features, i.e., the features before the last fully connected layer. Under the dual constraints of features angle and norm, MRAG can delicately encode the concatenation of cover and secret into subtle adversarial perturbations from local and global ranges relevant to steganalysis. Therefore, the resulting stego can achieve visual and steganalysis imperceptibility. Moreover, coarse-grained and fine-grained frequency decomposition operations are devised to transform the input, introducing multi-grained information. Extensive experiments demonstrate that MRAG can achieve state-of-the-art performance.

摘要: 图像隐藏充分探索了基于深度学习的模型的隐藏潜力，旨在隐藏封面图像中的图像级消息，并从隐写图像中揭示它们以实现隐蔽通信。由于覆盖类型仅限于空间域、单范围特征提取和攻击以及丢失约束不足，现有的隐藏方案很容易被肉眼或隐写分析器检测到。为了解决这些问题，我们提出了一种多范围表示驱动的对抗性隐写生成框架，称为MRAG，用于JPEG图像隐藏。这种设计源于这样一个事实：隐写分析器通常会结合局部范围和全球范围信息，以更好地捕获隐藏的痕迹。具体来说，MRAG集成了卷积的局部范围特性和Transformer的全局范围建模。与此同时，设计了特征角度规范解纠缠损失来发起多范围表示驱动的特征级对抗攻击。它根据代理隐写分析器的分类特征计算封面和隐写之间的对抗损失，即最后一个完全连接的层之前的特征。在特征角度和范数的双重约束下，MRAG可以将覆盖和秘密的级联精细地编码为与隐写分析相关的局部和全局范围的细微对抗性扰动。因此，所得到的隐写可以实现视觉和隐写分析的不可感知性。此外，设计了粗粒度和细粒度的频率分解操作来转换输入，引入多粒度信息。大量实验表明MRAG可以实现最先进的性能。



## **12. Untraceable DeepFakes via Traceable Fingerprint Elimination**

通过可追溯指纹消除不可追溯的DeepFakes cs.CR

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.03067v1) [paper-pdf](http://arxiv.org/pdf/2508.03067v1)

**Authors**: Jiewei Lai, Lan Zhang, Chen Tang, Pengcheng Sun, Xinming Wang, Yunhao Wang

**Abstract**: Recent advancements in DeepFakes attribution technologies have significantly enhanced forensic capabilities, enabling the extraction of traces left by generative models (GMs) in images, making DeepFakes traceable back to their source GMs. Meanwhile, several attacks have attempted to evade attribution models (AMs) for exploring their limitations, calling for more robust AMs. However, existing attacks fail to eliminate GMs' traces, thus can be mitigated by defensive measures. In this paper, we identify that untraceable DeepFakes can be achieved through a multiplicative attack, which can fundamentally eliminate GMs' traces, thereby evading AMs even enhanced with defensive measures. We design a universal and black-box attack method that trains an adversarial model solely using real data, applicable for various GMs and agnostic to AMs. Experimental results demonstrate the outstanding attack capability and universal applicability of our method, achieving an average attack success rate (ASR) of 97.08\% against 6 advanced AMs on DeepFakes generated by 9 GMs. Even in the presence of defensive mechanisms, our method maintains an ASR exceeding 72.39\%. Our work underscores the potential challenges posed by multiplicative attacks and highlights the need for more robust AMs.

摘要: DeepFakes归因技术的最新进展显着增强了取证能力，能够提取生成模型（GM）在图像中留下的痕迹，使DeepFakes可以追溯到其源GM。与此同时，一些攻击试图规避归因模型（AM）以探索其局限性，从而呼吁更强大的AM。然而，现有的攻击无法消除GM的痕迹，因此可以通过防御措施来缓解。在本文中，我们发现无法追踪的DeepFakes可以通过乘数攻击来实现，这可以从根本上消除GM的痕迹，从而避开AM，甚至通过防御措施增强。我们设计了一种通用的黑匣子攻击方法，该方法仅使用真实数据训练对抗模型，适用于各种GM且对AM不可知。实验结果表明，我们的方法具有出色的攻击能力和普遍适用性，针对9个GM生成的DeepFakes上的6个高级AM，平均攻击成功率（ASB）为97.08%。即使存在防御机制，我们的方法也能保持超过72.39%的ASB。我们的工作强调了多重攻击带来的潜在挑战，并强调了对更强大的AM的需求。



## **13. Attack Anything: Blind DNNs via Universal Background Adversarial Attack**

攻击一切：通过通用背景对抗攻击盲DNN cs.CV

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2409.00029v3) [paper-pdf](http://arxiv.org/pdf/2409.00029v3)

**Authors**: Jiawei Lian, Shaohui Mei, Xiaofei Wang, Yi Wang, Lefan Wang, Yingjie Lu, Mingyang Ma, Lap-Pui Chau

**Abstract**: It has been widely substantiated that deep neural networks (DNNs) are susceptible and vulnerable to adversarial perturbations. Existing studies mainly focus on performing attacks by corrupting targeted objects (physical attack) or images (digital attack), which is intuitively acceptable and understandable in terms of the attack's effectiveness. In contrast, our focus lies in conducting background adversarial attacks in both digital and physical domains, without causing any disruptions to the targeted objects themselves. Specifically, an effective background adversarial attack framework is proposed to attack anything, by which the attack efficacy generalizes well between diverse objects, models, and tasks. Technically, we approach the background adversarial attack as an iterative optimization problem, analogous to the process of DNN learning. Besides, we offer a theoretical demonstration of its convergence under a set of mild but sufficient conditions. To strengthen the attack efficacy and transferability, we propose a new ensemble strategy tailored for adversarial perturbations and introduce an improved smooth constraint for the seamless connection of integrated perturbations. We conduct comprehensive and rigorous experiments in both digital and physical domains across various objects, models, and tasks, demonstrating the effectiveness of attacking anything of the proposed method. The findings of this research substantiate the significant discrepancy between human and machine vision on the value of background variations, which play a far more critical role than previously recognized, necessitating a reevaluation of the robustness and reliability of DNNs. The code will be publicly available at https://github.com/JiaweiLian/Attack_Anything

摘要: 人们广泛证实，深度神经网络（DNN）容易受到对抗性扰动的影响。现有的研究主要集中在通过破坏目标对象（物理攻击）或图像（数字攻击）来进行攻击，就攻击的有效性而言，这是直观上可以接受的和可以理解的。相比之下，我们的重点是在数字和物理领域进行背景对抗攻击，而不会对目标对象本身造成任何干扰。具体来说，提出了一个有效的背景对抗攻击框架来攻击任何内容，通过该框架，攻击功效在不同的对象、模型和任务之间很好地推广。从技术上讲，我们将背景对抗攻击作为一个迭代优化问题来处理，类似于DNN学习的过程。此外，我们还在一组温和但充分的条件下对其收敛性进行了理论证明。为了加强攻击效率和可移植性，我们提出了一种针对对抗性扰动量身定制的新集成策略，并引入了改进的平滑约束来实现集成扰动的无缝连接。我们在数字和物理领域针对各种对象、模型和任务进行全面而严格的实验，证明攻击所提出方法的任何方法的有效性。这项研究的结果证实了人类和机器视觉在背景变化值上的显著差异，这比以前认识到的要重要得多，需要重新评估DNN的鲁棒性和可靠性。该代码将在https://github.com/JiaweiLian/Attack_Anything上公开发布



## **14. Long-tailed Adversarial Training with Self-Distillation**

自我蒸馏的长尾对抗训练 cs.CV

ICLR 2025. See OpenReview and code (in Supplementary Material) at  https://openreview.net/forum?id=vM94dZiqx4

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2503.06461v3) [paper-pdf](http://arxiv.org/pdf/2503.06461v3)

**Authors**: Seungju Cho, Hongsin Lee, Changick Kim

**Abstract**: Adversarial training significantly enhances adversarial robustness, yet superior performance is predominantly achieved on balanced datasets. Addressing adversarial robustness in the context of unbalanced or long-tailed distributions is considerably more challenging, mainly due to the scarcity of tail data instances. Previous research on adversarial robustness within long-tailed distributions has primarily focused on combining traditional long-tailed natural training with existing adversarial robustness methods. In this study, we provide an in-depth analysis for the challenge that adversarial training struggles to achieve high performance on tail classes in long-tailed distributions. Furthermore, we propose a simple yet effective solution to advance adversarial robustness on long-tailed distributions through a novel self-distillation technique. Specifically, this approach leverages a balanced self-teacher model, which is trained using a balanced dataset sampled from the original long-tailed dataset. Our extensive experiments demonstrate state-of-the-art performance in both clean and robust accuracy for long-tailed adversarial robustness, with significant improvements in tail class performance on various datasets. We improve the accuracy against PGD attacks for tail classes by 20.3, 7.1, and 3.8 percentage points on CIFAR-10, CIFAR-100, and Tiny-ImageNet, respectively, while achieving the highest robust accuracy.

摘要: 对抗训练显着增强了对抗稳健性，但卓越的性能主要是在平衡的数据集上实现的。在不平衡或长尾分布的背景下解决对抗稳健性更具挑战性，主要是由于尾部数据实例的稀缺性。之前关于长尾分布中对抗鲁棒性的研究主要集中在将传统的长尾自然训练与现有的对抗鲁棒性方法相结合。在这项研究中，我们对对抗训练难以在长尾分布中的尾类上实现高性能的挑战进行了深入分析。此外，我们提出了一种简单而有效的解决方案，通过一种新型的自蒸馏技术来提高长尾分布的对抗鲁棒性。具体来说，这种方法利用了平衡的自学模型，该模型使用从原始长尾数据集采样的平衡数据集进行训练。我们的广泛实验证明了长尾对抗鲁棒性在清晰和稳健的准确性方面具有最先进的性能，并且各种数据集的尾类性能也显着改进。我们将CIFAR-10、CIFAR-100和Tiny-ImageNet上针对尾部类别的PGD攻击的准确性分别提高了20.3、7.1和3.8个百分点，同时实现了最高的鲁棒准确性。



## **15. CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors**

CoCoTen：通过上下文共现张量的潜在空间特征检测大型语言模型的对抗性输入 cs.CL

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.02997v1) [paper-pdf](http://arxiv.org/pdf/2508.02997v1)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models. To support future research and reproducibility, we have made our implementation publicly available.

摘要: 大型语言模型（LLM）在许多应用中的广泛使用标志着研究和实践的重大进步。然而，它们的复杂性和难以理解的性质使它们容易受到攻击，尤其是旨在产生有害反应的越狱。为了应对这些威胁，开发强大的检测方法对于安全可靠地使用LLM至关重要。本文使用上下文共生矩阵来研究这个检测问题，该结构因其在数据稀缺环境中的有效性而被公认。我们提出了一种利用上下文同现矩阵和张量的潜在空间特征的新型方法，以有效识别对抗和越狱提示。我们的评估表明，这种方法仅使用0.5%的标记提示即可获得显着的0.83分，比基线提高了96.6%。这一结果凸显了我们所学习模式的力量，尤其是当标记数据稀缺时。与基线模型相比，我们的方法也明显更快，加速范围为2.3至128.4倍。为了支持未来的研究和可重复性，我们已公开我们的实施。



## **16. Adversarial Attention Perturbations for Large Object Detection Transformers**

大型物体检测变压器的对抗性注意力扰动 cs.CV

ICCV 2025

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2508.02987v1) [paper-pdf](http://arxiv.org/pdf/2508.02987v1)

**Authors**: Zachary Yahn, Selim Furkan Tekin, Fatih Ilhan, Sihao Hu, Tiansheng Huang, Yichang Xu, Margaret Loper, Ling Liu

**Abstract**: Adversarial perturbations are useful tools for exposing vulnerabilities in neural networks. Existing adversarial perturbation methods for object detection are either limited to attacking CNN-based detectors or weak against transformer-based detectors. This paper presents an Attention-Focused Offensive Gradient (AFOG) attack against object detection transformers. By design, AFOG is neural-architecture agnostic and effective for attacking both large transformer-based object detectors and conventional CNN-based detectors with a unified adversarial attention framework. This paper makes three original contributions. First, AFOG utilizes a learnable attention mechanism that focuses perturbations on vulnerable image regions in multi-box detection tasks, increasing performance over non-attention baselines by up to 30.6%. Second, AFOG's attack loss is formulated by integrating two types of feature loss through learnable attention updates with iterative injection of adversarial perturbations. Finally, AFOG is an efficient and stealthy adversarial perturbation method. It probes the weak spots of detection transformers by adding strategically generated and visually imperceptible perturbations which can cause well-trained object detection models to fail. Extensive experiments conducted with twelve large detection transformers on COCO demonstrate the efficacy of AFOG. Our empirical results also show that AFOG outperforms existing attacks on transformer-based and CNN-based object detectors by up to 83% with superior speed and imperceptibility. Code is available at https://github.com/zacharyyahn/AFOG.

摘要: 对抗性扰动是暴露神经网络漏洞的有用工具。现有的用于对象检测的对抗性扰动方法要么仅限于攻击基于CNN的检测器，要么对基于变压器的检测器弱。本文提出了针对对象检测转换器的注意力聚焦进攻梯度（AFOG）攻击。从设计上看，AFOG是神经架构不可知的，并且可以有效地攻击大型基于变压器的对象检测器和具有统一对抗性注意力框架的传统基于CNN的检测器。本文做出了三点原创贡献。首先，AFOG利用了一种可学习的注意力机制，该机制将扰动集中在多框检测任务中的脆弱图像区域上，使性能比非注意力基线提高了30.6%。其次，AFOG的攻击损失是通过可学习的注意力更新与对抗性扰动的迭代注入集成两种类型的特征损失来制定的。最后，AFOG是一种高效且隐蔽的对抗扰动方法。它通过添加战略性生成的且视觉上不可感知的扰动来探索检测转换器的弱点，这些扰动可能会导致训练有素的对象检测模型失败。在COCO上对十二个大型检测变压器进行的广泛实验证明了AFOG的功效。我们的经验结果还表明，AFOG比针对基于变压器和基于CNN的对象检测器的现有攻击性能高出83%，具有卓越的速度和不可感知性。代码可在https://github.com/zacharyyahn/AFOG上获取。



## **17. Augmented Adversarial Trigger Learning**

增强对抗触发学习 cs.LG

**SubmitDate**: 2025-08-05    [abs](http://arxiv.org/abs/2503.12339v2) [paper-pdf](http://arxiv.org/pdf/2503.12339v2)

**Authors**: Zhe Wang, Yanjun Qi

**Abstract**: Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs. We released our code \href{https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning}{here}.

摘要: 基于梯度优化的对抗攻击方法自动学习对抗触发器，以生成越狱提示或泄露系统提示。在这项工作中，我们仔细研究了对抗性触发学习的优化目标，并提出了ATLA：具有增强目标的对抗性触发学习。ATLA将之前研究使用的负对似然损失改进为加权损失公式，该公式鼓励学习的对抗触发因素对响应格式代币进行更多优化。这使得ATLA能够仅从一个查询-响应对中学习对抗触发器，并且学习到的触发器可以很好地推广到其他类似的查询。我们进一步设计了一个变体，以增加触发优化与辅助损失，抑制逃避反应。我们展示了如何使用ATLA来学习对抗性后缀、越狱LLM和提取隐藏的系统提示。从经验上讲，我们证明，ATLA始终优于当前最先进的技术，实现了近100%的成功攻击，同时需要减少80%的查询。ATLA学习的越狱后缀对看不见的查询具有很高的泛化能力，并可以很好地转移到新的LLM。我们发布了我们的代码\href{https：//github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning}{此处}。



## **18. Online Robust Multi-Agent Reinforcement Learning under Model Uncertainties**

模型不确定下的在线鲁棒多智能体强化学习 cs.LG

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02948v1) [paper-pdf](http://arxiv.org/pdf/2508.02948v1)

**Authors**: Zain Ulabedeen Farhat, Debamita Ghosh, George K. Atia, Yue Wang

**Abstract**: Well-trained multi-agent systems can fail when deployed in real-world environments due to model mismatches between the training and deployment environments, caused by environment uncertainties including noise or adversarial attacks. Distributionally Robust Markov Games (DRMGs) enhance system resilience by optimizing for worst-case performance over a defined set of environmental uncertainties. However, current methods are limited by their dependence on simulators or large offline datasets, which are often unavailable. This paper pioneers the study of online learning in DRMGs, where agents learn directly from environmental interactions without prior data. We introduce the {\it Robust Optimistic Nash Value Iteration (RONAVI)} algorithm and provide the first provable guarantees for this setting. Our theoretical analysis demonstrates that the algorithm achieves low regret and efficiently finds the optimal robust policy for uncertainty sets measured by Total Variation divergence and Kullback-Leibler divergence. These results establish a new, practical path toward developing truly robust multi-agent systems.

摘要: 经过良好训练的多智能体系统在部署到现实环境中时可能会失败，原因是训练和部署环境之间的模型不匹配，这是由环境不确定性（包括噪声或对抗性攻击）造成的。分布鲁棒马尔可夫博弈（DRMG）通过在一组定义的环境不确定性上优化最坏情况下的性能来增强系统的弹性。然而，目前的方法是有限的，它们依赖于模拟器或大型离线数据集，这往往是不可用的。本文开创了DRMG在线学习的研究，其中代理人直接从环境交互中学习，而无需先验数据。我们引入了{\it稳健乐观纳什值迭代（RONAVI）}算法，并为此设置提供了第一个可证明的保证。我们的理论分析表明，该算法实现了低遗憾，并有效地找到了由总变差偏差和Kullback-Leibler偏差衡量的不确定性集的最佳鲁棒策略。这些结果为开发真正强大的多智能体系统建立了一条新的、实用的途径。



## **19. LMDG: Advancing Lateral Movement Detection Through High-Fidelity Dataset Generation**

LMDG：通过高保真数据集生成推进横向运动检测 cs.CR

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02942v1) [paper-pdf](http://arxiv.org/pdf/2508.02942v1)

**Authors**: Anas Mabrouk, Mohamed Hatem, Mohammad Mamun, Sherif Saad

**Abstract**: Lateral Movement (LM) attacks continue to pose a significant threat to enterprise security, enabling adversaries to stealthily compromise critical assets. However, the development and evaluation of LM detection systems are impeded by the absence of realistic, well-labeled datasets. To address this gap, we propose LMDG, a reproducible and extensible framework for generating high-fidelity LM datasets. LMDG automates benign activity generation, multi-stage attack execution, and comprehensive labeling of system and network logs, dramatically reducing manual effort and enabling scalable dataset creation. A central contribution of LMDG is Process Tree Labeling, a novel agent-based technique that traces all malicious activity back to its origin with high precision. Unlike prior methods such as Injection Timing or Behavioral Profiling, Process Tree Labeling enables accurate, step-wise labeling of malicious log entries, correlating each with a specific attack step and MITRE ATT\&CK TTPs. To our knowledge, this is the first approach to support fine-grained labeling of multi-step attacks, providing critical context for detection models such as attack path reconstruction. We used LMDG to generate a 25-day dataset within a 25-VM enterprise environment containing 22 user accounts. The dataset includes 944 GB of host and network logs and embeds 35 multi-stage LM attacks, with malicious events comprising less than 1% of total activity, reflecting a realistic benign-to-malicious ratio for evaluating detection systems. LMDG-generated datasets improve upon existing ones by offering diverse LM attacks, up-to-date attack patterns, longer attack timeframes, comprehensive data sources, realistic network architectures, and more accurate labeling.

摘要: 横向移动（LM）攻击继续对企业安全构成重大威胁，使对手能够悄悄地危害关键资产。然而，LM检测系统的开发和评估受到缺乏现实的，良好标记的数据集的阻碍。为了解决这一差距，我们提出了LMDG，一个可复制和可扩展的框架，用于生成高保真LM数据集。LMDG可自动生成良性活动、执行多阶段攻击以及全面标记系统和网络日志，从而显著减少手动工作量并实现可扩展的数据集创建。LMDG的核心贡献是进程树标签，这是一种基于代理的新型技术，可以高精度地追踪所有恶意活动的起源。与注入计时或行为分析等先前的方法不同，进程树标签可以准确、逐步地标记恶意日志条目，并将每个条目与特定的攻击步骤和MITRE ATT\& CK TTP关联起来。据我们所知，这是第一种支持多步攻击细粒度标记的方法，为攻击路径重建等检测模型提供关键上下文。我们使用LMDG在包含22个用户帐户的25个虚拟机企业环境中生成25天数据集。该数据集包括944 GB的主机和网络日志，并嵌入35种多阶段LM攻击，其中恶意事件占总活动的不到1%，反映了评估检测系统的现实善意与恶意比率。LMDG生成的数据集通过提供多样化的LM攻击、最新的攻击模式、更长的攻击时间范围、全面的数据源、现实的网络架构和更准确的标签来改进现有数据集。



## **20. GRILL: Gradient Signal Restoration in Ill-Conditioned Layers to Enhance Adversarial Attacks on Autoencoders**

GRILL：病态层中的梯度信号恢复以增强对自动编码器的对抗攻击 cs.LG

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2505.03646v2) [paper-pdf](http://arxiv.org/pdf/2505.03646v2)

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Tobias Callies, Eirini Ntoutsi

**Abstract**: Adversarial robustness of deep autoencoders (AEs) remains relatively unexplored, even though their non-invertible nature poses distinct challenges. Existing attack algorithms during the optimization of imperceptible, norm-bounded adversarial perturbations to maximize output damage in AEs, often stop at sub-optimal attacks. We observe that the adversarial loss gradient vanishes when backpropagated through ill-conditioned layers. This issue arises from near-zero singular values in the Jacobians of these layers, which weaken the gradient signal during optimization. We introduce GRILL, a technique that locally restores gradient signals in ill-conditioned layers, enabling more effective norm-bounded attacks. Through extensive experiments on different architectures of popular AEs, under both sample-specific and universal attack setups, and across standard and adaptive attack settings, we show that our method significantly increases the effectiveness of our adversarial attacks, enabling a more rigorous evaluation of AE robustness.

摘要: 深度自动编码器（AE）的对抗鲁棒性仍然相对未被探索，尽管它们的不可逆性质带来了明显的挑战。现有的攻击算法在优化不可感知的、规范有界的对抗性扰动以最大化AE中的输出损害期间，通常停止在次优攻击。我们观察到，当反向传播穿过病态层时，对抗性损失梯度消失。这个问题源于这些层的雅可比量中接近零的奇异值，这会削弱优化期间的梯度信号。我们引入了GRILL，这是一种在病态层中局部恢复梯度信号的技术，从而实现更有效的规范有界攻击。通过在特定样本和通用攻击设置下以及标准和自适应攻击设置下对流行AE的不同架构进行广泛实验，我们表明我们的方法显着提高了对抗性攻击的有效性，从而能够更严格地评估AE稳健性。



## **21. Defending Against Knowledge Poisoning Attacks During Retrieval-Augmented Generation**

检索增强生成中知识中毒攻击的防御 cs.LG

Preprint for Submission

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02835v1) [paper-pdf](http://arxiv.org/pdf/2508.02835v1)

**Authors**: Kennedy Edemacu, Vinay M. Shashidhar, Micheal Tuape, Dan Abudu, Beakcheol Jang, Jong Wook Kim

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful approach to boost the capabilities of large language models (LLMs) by incorporating external, up-to-date knowledge sources. However, this introduces a potential vulnerability to knowledge poisoning attacks, where attackers can compromise the knowledge source to mislead the generation model. One such attack is the PoisonedRAG in which the injected adversarial texts steer the model to generate an attacker-chosen response to a target question. In this work, we propose novel defense methods, FilterRAG and ML-FilterRAG, to mitigate the PoisonedRAG attack. First, we propose a new property to uncover distinct properties to differentiate between adversarial and clean texts in the knowledge data source. Next, we employ this property to filter out adversarial texts from clean ones in the design of our proposed approaches. Evaluation of these methods using benchmark datasets demonstrate their effectiveness, with performances close to those of the original RAG systems.

摘要: 检索增强生成（RAG）已成为一种通过整合外部最新知识源来增强大型语言模型（LLM）能力的强大方法。然而，这会带来知识中毒攻击的潜在漏洞，攻击者可以损害知识源以误导生成模型。其中一种攻击是PoisonedRAG，其中注入的对抗性文本引导模型生成攻击者选择的对目标问题的响应。在这项工作中，我们提出了新颖的防御方法--FilterRAG和ML-FilterRAG，以减轻PoisonedRAG攻击。首先，我们提出了一个新的属性来揭示不同的属性，以区分知识数据源中的对抗性文本和干净文本。接下来，我们在设计我们提出的方法时利用这个属性从干净的文本中过滤出对抗性文本。使用基准数据集对这些方法进行评估，证明了它们的有效性，性能接近原始RAG系统。



## **22. Gandalf the Red: Adaptive Security for LLMs**

红色甘道夫：LLM的自适应安全 cs.LG

Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2501.07927v3) [paper-pdf](http://arxiv.org/pdf/2501.07927v3)

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Yun-Han Wu, Mateo Rojas-Carulla

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications.

摘要: 当前对大型语言模型（LLM）应用程序中针对即时攻击的防御的评估经常忽视两个关键因素：对抗行为的动态性质以及限制性防御对合法用户施加的可用性惩罚。我们提出了D-SEC（动态安全实用威胁模型），它明确地将攻击者与合法用户区分开来，对多步骤交互进行建模，并以可优化的形式表达安全实用。我们通过引入Gandalf来进一步解决现有评估中的缺陷，Gandalf是一个众包、游戏化的红色团队平台，旨在生成真实的、自适应的攻击。使用Gandalf，我们收集并发布了包含279，000次提示攻击的数据集。在良性用户数据的补充下，我们的分析揭示了安全性和实用性之间的相互作用，表明LLM中集成的防御（例如，系统提示）即使不阻止请求也会降低可用性。我们证明，限制应用程序域、深度防御和自适应防御是构建安全且有用的LLM应用程序的有效策略。



## **23. Adversarial flows: A gradient flow characterization of adversarial attacks**

对抗流：对抗攻击的梯度流特征 cs.LG

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2406.05376v3) [paper-pdf](http://arxiv.org/pdf/2406.05376v3)

**Authors**: Lukas Weigand, Tim Roith, Martin Burger

**Abstract**: A popular method to perform adversarial attacks on neuronal networks is the so-called fast gradient sign method and its iterative variant. In this paper, we interpret this method as an explicit Euler discretization of a differential inclusion, where we also show convergence of the discretization to the associated gradient flow. To do so, we consider the concept of p-curves of maximal slope in the case $p=\infty$. We prove existence of $\infty$-curves of maximum slope and derive an alternative characterization via differential inclusions. Furthermore, we also consider Wasserstein gradient flows for potential energies, where we show that curves in the Wasserstein space can be characterized by a representing measure on the space of curves in the underlying Banach space, which fulfill the differential inclusion. The application of our theory to the finite-dimensional setting is twofold: On the one hand, we show that a whole class of normalized gradient descent methods (in particular signed gradient descent) converge, up to subsequences, to the flow, when sending the step size to zero. On the other hand, in the distributional setting, we show that the inner optimization task of adversarial training objective can be characterized via $\infty$-curves of maximum slope on an appropriate optimal transport space.

摘要: 对神经元网络执行对抗攻击的一种流行方法是所谓的快速梯度符号法及其迭代变体。在本文中，我们将这种方法解释为微包含的显式欧拉离散化，其中我们还展示了离散化对相关梯度流的收敛性。为此，我们考虑在$p=\infty$的情况下最大斜坡p曲线的概念。我们证明了最大斜坡的$\infty$-曲线的存在性，并通过差异包含推导出替代特征。此外，我们还考虑了Wasserstein梯度流的势能，我们表明，曲线在Wasserstein空间可以通过表示的措施，在基础的Banach空间，这满足微分包含的曲线的空间。我们的理论的应用有限维设置是双重的：一方面，我们表明，一类正规化梯度下降方法（特别是有符号梯度下降）收敛，直到degreences，流，当发送的步长为零。另一方面，在分布式设置中，我们表明，内部优化任务的对抗训练目标可以通过$\infty$-曲线的最大斜率在适当的最佳运输空间。



## **24. Understanding the Risks of Asphalt Art on the Reliability of Surveillance Perception Systems**

了解沥青艺术对监控感知系统可靠性的风险 cs.CV

J. Ma and A. Enan are co-first authors; they have contributed  equally. This work has been submitted to the Transportation Research Record:  Journal of the Transportation Research Board for possible publication

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02530v1) [paper-pdf](http://arxiv.org/pdf/2508.02530v1)

**Authors**: Jin Ma, Abyad Enan, Long Cheng, Mashrur Chowdhury

**Abstract**: Artistic crosswalks featuring asphalt art, introduced by different organizations in recent years, aim to enhance the visibility and safety of pedestrians. However, their visual complexity may interfere with surveillance systems that rely on vision-based object detection models. In this study, we investigate the impact of asphalt art on pedestrian detection performance of a pretrained vision-based object detection model. We construct realistic crosswalk scenarios by compositing various street art patterns into a fixed surveillance scene and evaluate the model's performance in detecting pedestrians on asphalt-arted crosswalks under both benign and adversarial conditions. A benign case refers to pedestrian crosswalks painted with existing normal asphalt art, whereas an adversarial case involves digitally crafted or altered asphalt art perpetrated by an attacker. Our results show that while simple, color-based designs have minimal effect, complex artistic patterns, particularly those with high visual salience, can significantly degrade pedestrian detection performance. Furthermore, we demonstrate that adversarially crafted asphalt art can be exploited to deliberately obscure real pedestrians or generate non-existent pedestrian detections. These findings highlight a potential vulnerability in urban vision-based pedestrian surveillance systems and underscore the importance of accounting for environmental visual variations when designing robust pedestrian perception models.

摘要: 近年来，不同组织推出了以沥青艺术为特色的艺术人行横道，旨在提高行人的能见度和安全性。然而，它们的视觉复杂性可能会干扰依赖基于视觉的物体检测模型的监控系统。在这项研究中，我们研究了沥青艺术对预训练的基于视觉的对象检测模型的行人检测性能的影响。我们通过将各种街头艺术图案合成到固定的监控场景中来构建现实的人行横道场景，并评估该模型在良性和对抗条件下检测柏油艺术人行横道上行人的性能。良性案件指的是用现有的正常沥青艺术品绘制的人行横道，而对抗案件涉及攻击者使用数字方式制作或改变的沥青艺术品。我们的结果表明，虽然简单的基于颜色的设计效果很小，但复杂的艺术图案，尤其是具有高视觉显著性的图案，会显着降低行人检测性能。此外，我们证明，对抗性制作的沥青艺术可以被利用来故意模糊真实的行人或产生不存在的行人检测。这些发现凸显了城市基于视觉的行人监控系统的潜在弱点，并强调了在设计稳健的行人感知模型时考虑环境视觉变化的重要性。



## **25. Reference-free Adversarial Sex Obfuscation in Speech**

无参考的言语中的对抗性性别混淆 eess.AS

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02295v1) [paper-pdf](http://arxiv.org/pdf/2508.02295v1)

**Authors**: Yangyang Qu, Michele Panariello, Massimiliano Todisco, Nicholas Evans

**Abstract**: Sex conversion in speech involves privacy risks from data collection and often leaves residual sex-specific cues in outputs, even when target speaker references are unavailable. We introduce RASO for Reference-free Adversarial Sex Obfuscation. Innovations include a sex-conditional adversarial learning framework to disentangle linguistic content from sex-related acoustic markers and explicit regularisation to align fundamental frequency distributions and formant trajectories with sex-neutral characteristics learned from sex-balanced training data. RASO preserves linguistic content and, even when assessed under a semi-informed attack model, it significantly outperforms a competing approach to sex obfuscation.

摘要: 言语中的性别转换涉及数据收集的隐私风险，并且经常在输出中留下残留的特定性别线索，即使目标说话者参考不可用。我们引入RASO，用于无参考对抗性性别混淆。创新包括性别条件对抗学习框架，将语言内容与性别相关的声学标记区分开来，以及显式正规化，将基本频率分布和共振峰轨迹与从性别平衡训练数据中学习到的中性特征保持一致。RASO保留了语言内容，即使在半知情攻击模型下进行评估，它的表现也显着优于竞争性的性别混淆方法。



## **26. Two Heads are Better than One: Robust Learning Meets Multi-branch Models**

两个头脑总比一个头脑好：稳健学习满足多分支模型 cs.CV

10 pages, 5 Figures

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2208.08083v2) [paper-pdf](http://arxiv.org/pdf/2208.08083v2)

**Authors**: Zongyuan Zhang, Qingwen Bu, Tianyang Duan, Zheng Lin, Yuhao Qing, Zihan Fang, Heming Cui, Dong Huang

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples, in which DNNs are misled to false outputs due to inputs containing imperceptible perturbations. Adversarial training, a reliable and effective method of defense, may significantly reduce the vulnerability of neural networks and becomes the de facto standard for robust learning. While many recent works practice the data-centric philosophy, such as how to generate better adversarial examples or use generative models to produce additional training data, we look back to the models themselves and revisit the adversarial robustness from the perspective of deep feature distribution as an insightful complementarity. In this paper, we propose \textit{Branch Orthogonality adveRsarial Training} (BORT) to obtain state-of-the-art performance with solely the original dataset for adversarial training. To practice our design idea of integrating multiple orthogonal solution spaces, we leverage a simple and straightforward multi-branch neural network that eclipses adversarial attacks with no increase in inference time. We heuristically propose a corresponding loss function, branch-orthogonal loss, to make each solution space of the multi-branch model orthogonal. We evaluate our approach on CIFAR-10, CIFAR-100 and SVHN against $\ell_{\infty}$ norm-bounded perturbations of size $\epsilon = 8/255$, respectively. Exhaustive experiments are conducted to show that our method goes beyond all state-of-the-art methods without any tricks. Compared to all methods that do not use additional data for training, our models achieve 67.3\% and 41.5\% robust accuracy on CIFAR-10 and CIFAR-100 (improving upon the state-of-the-art by +7.23\% and +9.07\%). We also outperform methods using a training set with a far larger scale than ours.

摘要: 深度神经网络（DNN）容易受到对抗性示例的影响，其中DNN因包含不可感知的扰动的输入而被误导为错误输出。对抗训练是一种可靠有效的防御方法，可以显着降低神经网络的脆弱性，并成为稳健学习的事实标准。虽然最近的许多作品实践了以数据为中心的哲学，例如如何生成更好的对抗性示例或使用生成模型来生成额外的训练数据，但我们回顾了模型本身，并从深度特征分布的角度重新审视对抗性稳健性作为一种有洞察力的补充。在本文中，我们提议\textit{Branch Anonality adveRsarial Training}（BORT）仅利用对抗训练的原始数据集获得最先进的性能。为了实践我们集成多个垂直解空间的设计理念，我们利用了一个简单而直接的多分支神经网络，该网络在不增加推理时间的情况下超越了对抗性攻击。我们试探性地提出了一个相应的损失函数，即分支垂直损失，以使多分支模型的每个解空间垂直。我们分别针对大小为$\= 8/255$的$\ell_{\infty}$norm有界扰动评估了我们在CIFAR-10、CIFAR-100和SVHN上的方法。进行了详尽的实验，表明我们的方法超越了所有最先进的方法，没有任何技巧。与所有不使用额外数据进行训练的方法相比，我们的模型在CIFAR-10和CIFAR-100上实现了67.3%和41.5%的稳健准确性（比最新技术水平提高了+7.23%我们还优于使用规模远大于我们的训练集的方法。



## **27. Pigeon-SL: Robust Split Learning Framework for Edge Intelligence under Malicious Clients**

Pigeon-SL：恶意客户端下边缘智能的稳健拆分学习框架 cs.LG

13 pages, 14 figures

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02235v1) [paper-pdf](http://arxiv.org/pdf/2508.02235v1)

**Authors**: Sangjun Park, Tony Q. S. Quek, Hyowoon Seo

**Abstract**: Recent advances in split learning (SL) have established it as a promising framework for privacy-preserving, communication-efficient distributed learning at the network edge. However, SL's sequential update process is vulnerable to even a single malicious client, which can significantly degrade model accuracy. To address this, we introduce Pigeon-SL, a novel scheme grounded in the pigeonhole principle that guarantees at least one entirely honest cluster among M clients, even when up to N of them are adversarial. In each global round, the access point partitions the clients into N+1 clusters, trains each cluster independently via vanilla SL, and evaluates their validation losses on a shared dataset. Only the cluster with the lowest loss advances, thereby isolating and discarding malicious updates. We further enhance training and communication efficiency with Pigeon-SL+, which repeats training on the selected cluster to match the update throughput of standard SL. We validate the robustness and effectiveness of our approach under three representative attack models -- label flipping, activation and gradient manipulation -- demonstrating significant improvements in accuracy and resilience over baseline SL methods in future intelligent wireless networks.

摘要: 分离学习（SL）的最新进展已使其成为网络边缘保护隐私、通信高效的分布式学习的一个有前途的框架。然而，SL的顺序更新过程即使是单个恶意客户端也容易受到攻击，这可能会显着降低模型准确性。为了解决这个问题，我们引入了Pigeon-SL，这是一种基于鸽子洞原则的新颖方案，可以保证M个客户端中至少有一个完全诚实的集群，即使其中多达N个客户端是敌对的。在每个全球轮中，接入点将客户端划分为N+1个集群，通过vanilla SL独立训练每个集群，并评估它们在共享数据集上的验证损失。只有丢失最低的集群才会前进，从而隔离和丢弃恶意更新。我们使用Pigeon-SL+进一步提高了训练和通信效率，它在选定的集群上重复训练，以匹配标准SL的更新吞吐量。我们在三种代表性攻击模型（标签翻转、激活和梯度操纵）下验证了我们方法的稳健性和有效性，证明了未来智能无线网络中比基线SL方法在准确性和弹性方面的显着提高。



## **28. Failure Cases Are Better Learned But Boundary Says Sorry: Facilitating Smooth Perception Change for Accuracy-Robustness Trade-Off in Adversarial Training**

失败案例可以更好地学习，但边界说抱歉：促进认知的顺利改变，以实现对抗性训练中的准确性与稳健性权衡 cs.CV

2025 IEEE/CVF International Conference on Computer Vision (ICCV'25)

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02186v1) [paper-pdf](http://arxiv.org/pdf/2508.02186v1)

**Authors**: Yanyun Wang, Li Liu

**Abstract**: Adversarial Training (AT) is one of the most effective methods to train robust Deep Neural Networks (DNNs). However, AT creates an inherent trade-off between clean accuracy and adversarial robustness, which is commonly attributed to the more complicated decision boundary caused by the insufficient learning of hard adversarial samples. In this work, we reveal a counterintuitive fact for the first time: From the perspective of perception consistency, hard adversarial samples that can still attack the robust model after AT are already learned better than those successfully defended. Thus, different from previous views, we argue that it is rather the over-sufficient learning of hard adversarial samples that degrades the decision boundary and contributes to the trade-off problem. Specifically, the excessive pursuit of perception consistency would force the model to view the perturbations as noise and ignore the information within them, which should have been utilized to induce a smoother perception transition towards the decision boundary to support its establishment to an appropriate location. In response, we define a new AT objective named Robust Perception, encouraging the model perception to change smoothly with input perturbations, based on which we propose a novel Robust Perception Adversarial Training (RPAT) method, effectively mitigating the current accuracy-robustness trade-off. Experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet with ResNet-18, PreActResNet-18, and WideResNet-34-10 demonstrate the effectiveness of our method beyond four common baselines and 12 state-of-the-art (SOTA) works. The code is available at https://github.com/FlaAI/RPAT.

摘要: 对抗训练（AT）是训练稳健深度神经网络（DNN）的最有效方法之一。然而，AT在清晰的准确性和对抗稳健性之间创造了固有的权衡，这通常归因于硬对抗样本学习不足导致的更复杂的决策边界。在这项工作中，我们首次揭示了一个违反直觉的事实：从感知一致性的角度来看，AT后仍然可以攻击稳健模型的硬对抗样本已经比成功防御的样本学习得更好。因此，与以前的观点不同，我们认为，正是对硬对抗样本的过度学习降低了决策边界并导致了权衡问题。具体来说，过度追求感知一致性将迫使模型将扰动视为噪音并忽略其中的信息，这些信息本应用于诱导向决策边界更平稳的感知过渡，以支持其建立到适当的位置。作为回应，我们定义了一个名为“鲁棒感知”的新AT目标，鼓励模型感知随着输入扰动而平稳变化，在此基础上，我们提出了一种新型的鲁棒感知对抗训练（RMat）方法，有效地减轻了当前的准确性与鲁棒性的权衡。在CIFAR-10、CIFAR-100和Tiny-ImageNet上使用ResNet-18、PreActResNet-18和WideResNet-34-10进行实验，证明了我们的方法的有效性，超越了四个常见基线和12个最先进的（SOTA）作品。该代码可在https://github.com/FlaAI/RPAT上获取。



## **29. Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools**

吸引人的元数据攻击：诱导LLM代理攻击恶意工具 cs.AI

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.02110v1) [paper-pdf](http://arxiv.org/pdf/2508.02110v1)

**Authors**: Kanghua Mo, Li Hu, Yucheng Long, Zhihao Li

**Abstract**: Large language model (LLM) agents have demonstrated remarkable capabilities in complex reasoning and decision-making by leveraging external tools. However, this tool-centric paradigm introduces a previously underexplored attack surface: adversaries can manipulate tool metadata -- such as names, descriptions, and parameter schemas -- to influence agent behavior. We identify this as a new and stealthy threat surface that allows malicious tools to be preferentially selected by LLM agents, without requiring prompt injection or access to model internals. To demonstrate and exploit this vulnerability, we propose the Attractive Metadata Attack (AMA), a black-box in-context learning framework that generates highly attractive but syntactically and semantically valid tool metadata through iterative optimization. Our attack integrates seamlessly into standard tool ecosystems and requires no modification to the agent's execution framework. Extensive experiments across ten realistic, simulated tool-use scenarios and a range of popular LLM agents demonstrate consistently high attack success rates (81\%-95\%) and significant privacy leakage, with negligible impact on primary task execution. Moreover, the attack remains effective even under prompt-level defenses and structured tool-selection protocols such as the Model Context Protocol, revealing systemic vulnerabilities in current agent architectures. These findings reveal that metadata manipulation constitutes a potent and stealthy attack surface, highlighting the need for execution-level security mechanisms that go beyond prompt-level defenses.

摘要: 大型语言模型（LLM）代理通过利用外部工具在复杂推理和决策方面表现出了非凡的能力。然而，这种以工具为中心的范式引入了以前未充分研究的攻击表面：对手可以操纵工具元数据（例如名称、描述和参数模式）来影响代理行为。我们将其识别为一种新的隐形威胁表面，允许LLM代理优先选择恶意工具，而无需立即注入或访问模型内部。为了演示和利用此漏洞，我们提出了有吸引力的元数据攻击（AMA），这是一种黑匣子背景学习框架，通过迭代优化生成高度有吸引力但在语法和语义上有效的工具元数据。我们的攻击无缝集成到标准工具生态系统中，并且不需要修改代理的执行框架。针对十个真实的模拟工具使用场景和一系列流行的LLM代理的广泛实验表明，攻击成功率始终很高（81%-95%）和严重的隐私泄露，对主要任务执行的影响可以忽略不计。此外，即使在预算级防御和结构化工具选择协议（例如模型上下文协议）下，该攻击仍然有效，从而揭示了当前代理体系结构中的系统性漏洞。这些发现表明，元数据操纵构成了一种强大且隐蔽的攻击表面，凸显了对超出预算级别防御的执行级别安全机制的需求。



## **30. Controllable and Stealthy Shilling Attacks via Dispersive Latent Diffusion**

通过分散潜在扩散进行可控且隐形的先令攻击 cs.LG

**SubmitDate**: 2025-08-04    [abs](http://arxiv.org/abs/2508.01987v1) [paper-pdf](http://arxiv.org/pdf/2508.01987v1)

**Authors**: Shutong Qiao, Wei Yuan, Junliang Yu, Tong Chen, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Recommender systems (RSs) are now fundamental to various online platforms, but their dependence on user-contributed data leaves them vulnerable to shilling attacks that can manipulate item rankings by injecting fake users. Although widely studied, most existing attack models fail to meet two critical objectives simultaneously: achieving strong adversarial promotion of target items while maintaining realistic behavior to evade detection. As a result, the true severity of shilling threats that manage to reconcile the two objectives remains underappreciated. To expose this overlooked vulnerability, we present DLDA, a diffusion-based attack framework that can generate highly effective yet indistinguishable fake users by enabling fine-grained control over target promotion. Specifically, DLDA operates in a pre-aligned collaborative embedding space, where it employs a conditional latent diffusion process to iteratively synthesize fake user profiles with precise target item control. To evade detection, DLDA introduces a dispersive regularization mechanism that promotes variability and realism in generated behavioral patterns. Extensive experiments on three real-world datasets and five popular RS models demonstrate that, compared to prior attacks, DLDA consistently achieves stronger item promotion while remaining harder to detect. These results highlight that modern RSs are more vulnerable than previously recognized, underscoring the urgent need for more robust defenses.

摘要: 推荐系统（RS）现在是各种在线平台的基础，但它们对用户贡献的数据的依赖使它们容易受到先令攻击，这些攻击可以通过注入虚假用户来操纵物品排名。尽管经过广泛研究，但大多数现有的攻击模型未能同时满足两个关键目标：实现目标项的强对抗性推广，同时保持现实行为以逃避检测。因此，能够调和这两个目标的先令威胁的真正严重性仍然没有得到充分的认识。为了揭露这个被忽视的漏洞，我们提出了DLDA，这是一种基于扩散的攻击框架，可以通过对目标促销进行细粒度控制来生成高效但难以区分的虚假用户。具体来说，DLDA在预对齐的协作嵌入空间中运行，其中它采用条件潜在扩散过程来迭代合成虚假用户配置文件，并具有精确的目标项控制。为了逃避检测，DLDA引入了分散的规则化机制，以促进生成的行为模式的可变性和真实性。对三个现实世界数据集和五个流行的RS模型的广泛实验表明，与之前的攻击相比，DLDA始终实现了更强的物品促销，同时仍然更难检测。这些结果凸显了现代RS比以前认识到的更脆弱，凸显了对更强大防御的迫切需要。



## **31. Proactive Disentangled Modeling of Trigger-Object Pairings for Backdoor Defense**

后门防御触发-对象配对的主动解开建模 cs.CV

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01932v1) [paper-pdf](http://arxiv.org/pdf/2508.01932v1)

**Authors**: Kyle Stein, Andrew A. Mahyari, Guillermo Francia III, Eman El-Sheikh

**Abstract**: Deep neural networks (DNNs) and generative AI (GenAI) are increasingly vulnerable to backdoor attacks, where adversaries embed triggers into inputs to cause models to misclassify or misinterpret target labels. Beyond traditional single-trigger scenarios, attackers may inject multiple triggers across various object classes, forming unseen backdoor-object configurations that evade standard detection pipelines. In this paper, we introduce DBOM (Disentangled Backdoor-Object Modeling), a proactive framework that leverages structured disentanglement to identify and neutralize both seen and unseen backdoor threats at the dataset level. Specifically, DBOM factorizes input image representations by modeling triggers and objects as independent primitives in the embedding space through the use of Vision-Language Models (VLMs). By leveraging the frozen, pre-trained encoders of VLMs, our approach decomposes the latent representations into distinct components through a learnable visual prompt repository and prompt prefix tuning, ensuring that the relationships between triggers and objects are explicitly captured. To separate trigger and object representations in the visual prompt repository, we introduce the trigger-object separation and diversity losses that aids in disentangling trigger and object visual features. Next, by aligning image features with feature decomposition and fusion, as well as learned contextual prompt tokens in a shared multimodal space, DBOM enables zero-shot generalization to novel trigger-object pairings that were unseen during training, thereby offering deeper insights into adversarial attack patterns. Experimental results on CIFAR-10 and GTSRB demonstrate that DBOM robustly detects poisoned images prior to downstream training, significantly enhancing the security of DNN training pipelines.

摘要: 深度神经网络（DNN）和生成人工智能（GenAI）越来越容易受到后门攻击，对手将触发器嵌入到输入中，导致模型错误分类或误解目标标签。除了传统的单触发器场景之外，攻击者可能会在各种对象类中注入多个触发器，形成逃避标准检测管道的看不见的后门对象配置。在本文中，我们介绍了DBOM（分离后门对象建模），这是一个积极主动的框架，利用结构化分离来识别和抵消数据集级别上可见和不可见的后门威胁。具体来说，DBMI通过使用视觉语言模型（VLM）将触发器和对象建模为嵌入空间中的独立基元，对输入图像表示进行因式分解。通过利用冻结的、预先训练的VLM编码器，我们的方法通过可学习的视觉提示存储库和提示前缀调整将潜在表示分解为不同的组件，确保明确捕获触发器和对象之间的关系。为了在视觉提示库中分离触发器和对象表示，我们引入了触发器对象分离和多样性损失，这有助于解开触发器和对象的视觉特征。接下来，通过将图像特征与特征分解和融合以及共享多模态空间中学习的上下文提示标记进行对齐，DBOM能够对训练期间看不到的新型攻击者-对象配对进行零射击泛化，从而更深入地了解对抗性攻击模式。在CIFAR-10和GTSRB上的实验结果表明，DBOM在下游训练之前鲁棒地检测出中毒图像，显著增强了DNN训练管道的安全性。



## **32. Continual Adversarial Defense**

持续对抗性辩护 cs.CV

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2312.09481v6) [paper-pdf](http://arxiv.org/pdf/2312.09481v6)

**Authors**: Qian Wang, Hefei Ling, Yingwei Li, Qihao Liu, Ruoxi Jia, Ning Yu

**Abstract**: In response to the rapidly evolving nature of adversarial attacks against visual classifiers, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is unrealistic, as the environment in which the defense system operates is dynamic. Over time, new attacks inevitably emerge that exploit the vulnerabilities of existing defenses and bypass them. Therefore, we propose a continual defense strategy under a practical threat model and, for the first time, introduce the Continual Adversarial Defense (CAD) framework. CAD continuously collects adversarial data online and adapts to evolving attack sequences, while adhering to four practical principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high classification accuracy on both clean and adversarial data. We explore and integrate cutting-edge techniques from continual learning, few-shot learning, and ensemble learning to fulfill the principles. Extensive experiments validate the effectiveness of our approach against multi-stage adversarial attacks and demonstrate significant improvements over a wide range of baseline methods. We further observe that CAD's defense performance tends to saturate as the number of attacks increases, indicating its potential as a persistent defense once adapted to a sufficiently diverse set of attacks. Our research sheds light on a brand-new paradigm for continual defense adaptation against dynamic and evolving attacks.

摘要: 为了应对针对视觉分类器的对抗性攻击的快速演变性质，人们提出了多种防御措施来概括针对尽可能多的已知攻击。然而，设计一种适用于所有类型攻击的防御方法是不现实的，因为防御系统运行的环境是动态的。随着时间的推移，新的攻击不可避免地会出现，这些攻击会利用现有防御系统的漏洞并绕过它们。因此，我们在实际威胁模型下提出了持续防御策略，并首次引入了持续对抗防御（CAD）框架。CAD持续在线收集对抗数据并适应不断发展的攻击序列，同时遵循四项实用原则：（1）持续适应新攻击，而不会发生灾难性遗忘，（2）少量攻击适应，（3）内存高效适应，（4）干净和对抗数据的高分类准确性。我们探索并整合来自持续学习、少量学习和综合学习的尖端技术以实现这些原则。大量的实验验证了我们的方法对抗多阶段对抗攻击的有效性，并证明了在广泛的基线方法上的显着改进。我们进一步观察到，随着攻击数量的增加，CAD的防御性能往往会饱和，这表明一旦适应足够多样化的攻击，它就有潜力作为持久防御。我们的研究揭示了一种全新的范式，用于针对动态和不断发展的攻击进行持续防御适应。



## **33. Beyond Vulnerabilities: A Survey of Adversarial Attacks as Both Threats and Defenses in Computer Vision Systems**

超越漏洞：对抗性攻击作为计算机视觉系统威胁和防御的调查 cs.CV

33 pages

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01845v1) [paper-pdf](http://arxiv.org/pdf/2508.01845v1)

**Authors**: Zhongliang Guo, Yifei Qian, Yanli Li, Weiye Li, Chun Tong Lei, Shuai Zhao, Lei Fang, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Adversarial attacks against computer vision systems have emerged as a critical research area that challenges the fundamental assumptions about neural network robustness and security. This comprehensive survey examines the evolving landscape of adversarial techniques, revealing their dual nature as both sophisticated security threats and valuable defensive tools. We provide a systematic analysis of adversarial attack methodologies across three primary domains: pixel-space attacks, physically realizable attacks, and latent-space attacks. Our investigation traces the technical evolution from early gradient-based methods such as FGSM and PGD to sophisticated optimization techniques incorporating momentum, adaptive step sizes, and advanced transferability mechanisms. We examine how physically realizable attacks have successfully bridged the gap between digital vulnerabilities and real-world threats through adversarial patches, 3D textures, and dynamic optical perturbations. Additionally, we explore the emergence of latent-space attacks that leverage semantic structure in internal representations to create more transferable and meaningful adversarial examples. Beyond traditional offensive applications, we investigate the constructive use of adversarial techniques for vulnerability assessment in biometric authentication systems and protection against malicious generative models. Our analysis reveals critical research gaps, particularly in neural style transfer protection and computational efficiency requirements. This survey contributes a comprehensive taxonomy, evolution analysis, and identification of future research directions, aiming to advance understanding of adversarial vulnerabilities and inform the development of more robust and trustworthy computer vision systems.

摘要: 针对计算机视觉系统的对抗攻击已成为一个关键研究领域，挑战了有关神经网络稳健性和安全性的基本假设。这项全面的调查探讨了对抗性技术的不断发展，揭示了它们的双重性质，既是复杂的安全威胁，又是宝贵的防御工具。我们对三个主要领域的对抗攻击方法进行了系统分析：像素空间攻击、物理可实现攻击和潜伏空间攻击。我们的研究追踪了从早期基于梯度的方法（例如FGSM和PVD）到结合动量、自适应步骤和高级可移植性机制的复杂优化技术的技术演变。我们研究了物理上可实现的攻击如何通过对抗补丁，3D纹理和动态光学扰动成功地弥合了数字漏洞和现实威胁之间的差距。此外，我们还探讨了潜在空间攻击的出现，这些攻击利用内部表示中的语义结构来创建更具可转移性和有意义的对抗性示例。除了传统的攻击性应用，我们还研究了对抗性技术在生物特征认证系统中的脆弱性评估和恶意生成模型的保护方面的建设性使用。我们的分析揭示了关键的研究差距，特别是在神经风格转移保护和计算效率的要求。该调查提供了全面的分类、进化分析和未来研究方向的确定，旨在促进对对抗性漏洞的理解，并为开发更强大、更值得信赖的计算机视觉系统提供信息。



## **34. SHIELD: Secure Hypernetworks for Incremental Expansion Learning Defense**

SHIELD：用于增量扩展学习防御的安全超网络 cs.LG

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2506.08255v2) [paper-pdf](http://arxiv.org/pdf/2506.08255v2)

**Authors**: Patryk Krukowski, Łukasz Gorczyca, Piotr Helm, Kamil Książek, Przemysław Spurek

**Abstract**: Continual learning under adversarial conditions remains an open problem, as existing methods often compromise either robustness, scalability, or both. We propose a novel framework that integrates Interval Bound Propagation (IBP) with a hypernetwork-based architecture to enable certifiably robust continual learning across sequential tasks. Our method, SHIELD, generates task-specific model parameters via a shared hypernetwork conditioned solely on compact task embeddings, eliminating the need for replay buffers or full model copies and enabling efficient over time. To further enhance robustness, we introduce Interval MixUp, a novel training strategy that blends virtual examples represented as $\ell_{\infty}$ balls centered around MixUp points. Leveraging interval arithmetic, this technique guarantees certified robustness while mitigating the wrapping effect, resulting in smoother decision boundaries. We evaluate SHIELD under strong white-box adversarial attacks, including PGD and AutoAttack, across multiple benchmarks. It consistently outperforms existing robust continual learning methods, achieving state-of-the-art average accuracy while maintaining both scalability and certification. These results represent a significant step toward practical and theoretically grounded continual learning in adversarial settings.

摘要: 对抗条件下的持续学习仍然是一个悬而未决的问题，因为现有的方法经常损害稳健性、可扩展性或两者兼而有之。我们提出了一种新颖的框架，将区间束缚传播（IPP）与基于超网络的架构集成，以实现跨顺序任务的可认证稳健的持续学习。我们的方法SHIELD通过仅以紧凑任务嵌入为条件的共享超网络生成特定于任务的模型参数，消除了对重播缓冲区或完整模型副本的需要，并随着时间的推移实现高效。为了进一步增强稳健性，我们引入了Interval MixUp，这是一种新颖的训练策略，它混合以MixUp点为中心的$\ell_{\infty}$球表示的虚拟示例。利用区间算术，该技术保证了经过认证的鲁棒性，同时减轻了包裹效应，从而获得更平滑的决策边界。我们在多个基准测试中评估SHIELD在强白盒对抗攻击（包括PVD和AutoAttack）下。它始终优于现有的强大持续学习方法，实现最先进的平均准确性，同时保持可扩展性和认证。这些结果代表着在对抗环境中朝着实践和理论基础的持续学习迈出了重要一步。



## **35. Enhancing Spectrogram Realism in Singing Voice Synthesis via Explicit Bandwidth Extension Prior to Vocoder**

通过声码器之前的显式带宽扩展增强歌唱声音合成中的频谱图现实性 cs.SD

7 pages, 8 figures

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01796v1) [paper-pdf](http://arxiv.org/pdf/2508.01796v1)

**Authors**: Runxuan Yang, Kai Li, Guo Chen, Xiaolin Hu

**Abstract**: This paper addresses the challenge of enhancing the realism of vocoder-generated singing voice audio by mitigating the distinguishable disparities between synthetic and real-life recordings, particularly in high-frequency spectrogram components. Our proposed approach combines two innovations: an explicit linear spectrogram estimation step using denoising diffusion process with DiT-based neural network architecture optimized for time-frequency data, and a redesigned vocoder based on Vocos specialized in handling large linear spectrograms with increased frequency bins. This integrated method can produce audio with high-fidelity spectrograms that are challenging for both human listeners and machine classifiers to differentiate from authentic recordings. Objective and subjective evaluations demonstrate that our streamlined approach maintains high audio quality while achieving this realism. This work presents a substantial advancement in overcoming the limitations of current vocoding techniques, particularly in the context of adversarial attacks on fake spectrogram detection.

摘要: 本文通过减轻合成录音和现实生活录音之间的明显差异，特别是在高频频谱图成分中，解决了增强声码器生成的歌唱声音音频真实感的挑战。我们提出的方法结合了两个创新：使用去噪扩散过程的显式线性谱图估计步骤，并针对时频数据优化的基于DiT的神经网络架构，以及基于Vocos的重新设计的声码器，专门处理具有增加频率范围的大型线性谱图。这种集成方法可以产生具有高保真度频谱图的音频，这对于人类听众和机器分类器来说都是一个挑战，难以区分真实录音。客观和主观评估表明，我们的简化方法可以在实现真实感的同时保持高音频质量。这项工作在克服当前声码技术的局限性方面取得了重大进步，特别是在对假谱图检测的对抗性攻击的背景下。



## **36. "Energon": Unveiling Transformers from GPU Power and Thermal Side-Channels**

“Energon”：推出来自图形处理器电源和散热侧通道的变形金刚 cs.CR

Accepted at IEEE/ACM International Conference on Computer-Aided  Design, 2025

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01768v1) [paper-pdf](http://arxiv.org/pdf/2508.01768v1)

**Authors**: Arunava Chaudhuri, Shubhi Shukla, Sarani Bhattacharya, Debdeep Mukhopadhyay

**Abstract**: Transformers have become the backbone of many Machine Learning (ML) applications, including language translation, summarization, and computer vision. As these models are increasingly deployed in shared Graphics Processing Unit (GPU) environments via Machine Learning as a Service (MLaaS), concerns around their security grow. In particular, the risk of side-channel attacks that reveal architectural details without physical access remains underexplored, despite the high value of the proprietary models they target. This work to the best of our knowledge is the first to investigate GPU power and thermal fluctuations as side-channels and further exploit them to extract information from pre-trained transformer models. The proposed analysis shows how these side channels can be exploited at user-privilege to reveal critical architectural details such as encoder/decoder layer and attention head for both language and vision transformers. We demonstrate the practical impact by evaluating multiple language and vision pre-trained transformers which are publicly available. Through extensive experimental evaluations, we demonstrate that the attack model achieves a high accuracy of over 89% on average for model family identification and 100% for hyperparameter classification, in both single-process as well as noisy multi-process scenarios. Moreover, by leveraging the extracted architectural information, we demonstrate highly effective black-box transfer adversarial attacks with an average success rate exceeding 93%, underscoring the security risks posed by GPU side-channel leakage in deployed transformer models.

摘要: Transformer已经成为许多机器学习（ML）应用程序的支柱，包括语言翻译，摘要和计算机视觉。随着这些模型越来越多地通过机器学习即服务（MLaaS）部署在共享图形处理单元（GPU）环境中，对其安全性的担忧也在增长。特别是，侧信道攻击的风险，揭示了没有物理访问的架构细节仍然没有得到充分的研究，尽管他们的目标专有模型的高价值。据我们所知，这项工作是第一次将图形处理器功率和热波动作为侧通道进行研究，并进一步利用它们从预训练的Transformer模型中提取信息。所提出的分析显示了如何利用这些侧通道在用户权限，揭示关键的架构细节，如编码器/解码器层和注意力头的语言和视觉转换器。我们通过评估公开提供的多语言和视觉预训练变压器来展示实际影响。通过广泛的实验评估，我们证明了攻击模型在单进程和有噪声的多进程场景中，平均模型族识别和超参数分类的准确率均超过89%。此外，通过利用提取的架构信息，我们展示了高效的黑匣子传输对抗攻击，平均成功率超过93%，强调了部署的Transformer模型中的图形处理器侧通道泄漏所带来的安全风险。



## **37. Optimal and Feasible Contextuality-based Randomness Generation**

最佳可行的基于上下文的随机生成 quant-ph

Accepted in Phys. Rev. Lett. 7+17 pages, 2+5 figures

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2412.20126v2) [paper-pdf](http://arxiv.org/pdf/2412.20126v2)

**Authors**: Yuan Liu, Ravishankar Ramanathan

**Abstract**: Semi-device-independent (SDI) randomness generation protocols based on Kochen-Specker contextuality offer the attractive features of compact devices, high rates, and ease of experimental implementation over fully device-independent (DI) protocols. Here, we investigate this paradigm and derive four results to improve the state-of-art. Firstly, we introduce a family of simple, experimentally feasible orthogonality graphs (measurement compatibility structures) for which the maximum violation of the corresponding non-contextuality inequalities allows to certify the maximum amount of $\log_2 d$ bits of randomness from a qu$d$it system with projective measurements for $d \geq 3$. We analytically derive the Lov\'asz theta and fractional packing number for this graph family, and thereby prove their utility for optimal randomness generation in both randomness expansion and amplification tasks. Secondly, a central additional assumption in contextuality-based protocols over fully DI ones, is that the measurements are repeatable and satisfy an intended compatibility structure. We frame a relaxation of this condition in terms of $\epsilon$-orthogonality graphs for a parameter $\epsilon > 0$, and derive quantum correlations that allow to certify randomness for arbitrary relaxation $\epsilon \in [0,1)$. Thirdly, it is well known that a single qubit is non-contextual, i.e., the qubit correlations can be explained by a non-contextual hidden variable (NCHV) model. We show however that a single qubit is \textit{almost} contextual, in that there exist qubit correlations that cannot be explained by $\epsilon$-faithful NCHV models for small $\epsilon > 0$. Finally, we point out possible attacks by quantum and general consistent (non-signalling) adversaries for certain classes of contextuality tests over and above those considered in DI scenarios.

摘要: 基于Kochen-Specker上下文的半设备独立（SDF）随机性生成协议提供了紧凑设备、高速率以及比完全设备独立（DI）协议易于实验实施的吸引力特征。在这里，我们研究了这个范式并得出了四个结果来改进最新技术水平。首先，我们引入了一系列简单的、实验上可行的正向图（测量兼容性结构），对于这些图，相应的非上下文不等式的最大破坏允许证明具有$d \geq 3$的投影测量的qu$d$it系统的最大$\log_2 d$比特随机性。我们通过分析推导出该图族的Lov ' asz theta和分数包装数，从而证明它们对于随机性扩展和放大任务中的最佳随机性生成的实用性。其次，与完全DI协议相比，基于上下文的协议中的一个核心附加假设是测量是可重复的并且满足预期的兼容性结构。我们根据参数$\> 0$的$\$-正交图来定义该条件的松弛，并推导出量子相关性，允许证明任意松弛$\\in [0，1）$的随机性。第三，众所周知，单个量子位是非上下文的，即，量子位相关性可以通过非上下文隐藏变量（NCHV）模型来解释。然而，我们表明，单个量子位是\textit{almost}上下文的，因为存在着无法用$\$-忠实的NCHV模型解释的量子位相关性（对于小$\> 0$）。最后，我们指出量子和一般一致（非信号）对手可能对某些类别的上下文测试进行攻击，超出DI场景中考虑的攻击。



## **38. AI-Generated Text is Non-Stationary: Detection via Temporal Tomography**

人工智能生成的文本是非静止的：通过时间断层扫描检测 cs.CL

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01754v1) [paper-pdf](http://arxiv.org/pdf/2508.01754v1)

**Authors**: Alva West, Yixuan Weng, Minjun Zhu, Luodan Zhang, Zhen Lin, Guangsheng Bao, Yue Zhang

**Abstract**: The field of AI-generated text detection has evolved from supervised classification to zero-shot statistical analysis. However, current approaches share a fundamental limitation: they aggregate token-level measurements into scalar scores, discarding positional information about where anomalies occur. Our empirical analysis reveals that AI-generated text exhibits significant non-stationarity, statistical properties vary by 73.8\% more between text segments compared to human writing. This discovery explains why existing detectors fail against localized adversarial perturbations that exploit this overlooked characteristic. We introduce Temporal Discrepancy Tomography (TDT), a novel detection paradigm that preserves positional information by reformulating detection as a signal processing task. TDT treats token-level discrepancies as a time-series signal and applies Continuous Wavelet Transform to generate a two-dimensional time-scale representation, capturing both the location and linguistic scale of statistical anomalies. On the RAID benchmark, TDT achieves 0.855 AUROC (7.1\% improvement over the best baseline). More importantly, TDT demonstrates robust performance on adversarial tasks, with 14.1\% AUROC improvement on HART Level 2 paraphrasing attacks. Despite its sophisticated analysis, TDT maintains practical efficiency with only 13\% computational overhead. Our work establishes non-stationarity as a fundamental characteristic of AI-generated text and demonstrates that preserving temporal dynamics is essential for robust detection.

摘要: 人工智能生成的文本检测领域已经从监督分类发展到零镜头统计分析。然而，当前的方法都有一个根本性的局限性：它们将标记级测量结果聚合为纯量分数，丢弃有关异常发生位置的位置信息。我们的实证分析表明，人工智能生成的文本表现出显着的非平稳性，与人类写作相比，文本片段之间的统计属性差异大73.8%。这一发现解释了为什么现有的检测器无法对抗利用这一被忽视的特征的局部对抗性扰动。我们介绍了时间离散断层扫描（TDT），一种新的检测范式，保留位置信息，通过重新制定检测作为一个信号处理任务。TDT将标记级差异视为时间序列信号，并应用连续小波变换生成二维时间尺度表示，捕获统计异常的位置和语言尺度。在RAID基准测试中，TDT达到0.855 AUROC（比最佳基线提高了7.1%）。更重要的是，TDT在对抗性任务上表现出了强大的性能，在HART 2级释义攻击上有14.1\%的AUROC改进。尽管TDT分析复杂，但只需13%的计算费用即可保持实际效率。我们的工作将非平稳性确立为人工智能生成文本的基本特征，并证明保留时间动态对于鲁棒检测至关重要。



## **39. Simulated Ensemble Attack: Transferring Jailbreaks Across Fine-tuned Vision-Language Models**

模拟集群攻击：通过微调的视觉语言模型转移越狱 cs.CV

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01741v1) [paper-pdf](http://arxiv.org/pdf/2508.01741v1)

**Authors**: Ruofan Wang, Xin Wang, Yang Yao, Xuan Tong, Xingjun Ma

**Abstract**: Fine-tuning open-source Vision-Language Models (VLMs) creates a critical yet underexplored attack surface: vulnerabilities in the base VLM could be retained in fine-tuned variants, rendering them susceptible to transferable jailbreak attacks. To demonstrate this risk, we introduce the Simulated Ensemble Attack (SEA), a novel grey-box jailbreak method in which the adversary has full access to the base VLM but no knowledge of the fine-tuned target's weights or training configuration. To improve jailbreak transferability across fine-tuned VLMs, SEA combines two key techniques: Fine-tuning Trajectory Simulation (FTS) and Targeted Prompt Guidance (TPG). FTS generates transferable adversarial images by simulating the vision encoder's parameter shifts, while TPG is a textual strategy that steers the language decoder toward adversarially optimized outputs. Experiments on the Qwen2-VL family (2B and 7B) demonstrate that SEA achieves high transfer attack success rates exceeding 86.5% and toxicity rates near 49.5% across diverse fine-tuned variants, even those specifically fine-tuned to improve safety behaviors. Notably, while direct PGD-based image jailbreaks rarely transfer across fine-tuned VLMs, SEA reliably exploits inherited vulnerabilities from the base model, significantly enhancing transferability. These findings highlight an urgent need to safeguard fine-tuned proprietary VLMs against transferable vulnerabilities inherited from open-source foundations, motivating the development of holistic defenses across the entire model lifecycle.

摘要: 微调开源视觉语言模型（VLM）创建了一个关键但未充分探索的攻击面：基础VLM中的漏洞可能会保留在微调的变体中，使它们容易受到可转移的越狱攻击。为了证明这种风险，我们引入了模拟包围攻击（SEA），一种新的灰盒越狱方法，其中对手可以完全访问基础VLM，但不知道微调目标的权重或训练配置。为了提高在微调VLM之间的越狱可转移性，SEA结合了两个关键技术：微调轨迹模拟（FTS）和目标提示制导（TPG）。FTS通过模拟视觉编码器的参数变化来生成可转移的对抗图像，而TPG是一种文本策略，可以将语言解码器引导到对抗优化的输出。Qwen 2-BL家族（2B和7 B）的实验表明，SEA在各种微调变体中实现了超过86.5%的高转移攻击成功率和接近49.5%的毒性率，即使是那些专门微调以改善安全行为的变体。值得注意的是，虽然直接基于PGD的图像越狱很少通过微调的VLM传输，但SEA可靠地利用了从基本模型继承的漏洞，显着增强了可传输性。这些发现凸显了迫切需要保护微调的专有VLM免受从开源基金会继承的可转移漏洞的影响，从而激励整个模型生命周期中的整体防御开发。



## **40. RedDiffuser: Red Teaming Vision-Language Models for Toxic Continuation via Reinforced Stable Diffusion**

RedDiffuser：通过增强稳定扩散的有毒延续的红色团队视觉语言模型 cs.CV

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2503.06223v3) [paper-pdf](http://arxiv.org/pdf/2503.06223v3)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: Vision-Language Models (VLMs) are vulnerable to jailbreak attacks, where adversaries bypass safety mechanisms to elicit harmful outputs. In this work, we examine an insidious variant of this threat: toxic continuation. Unlike standard jailbreaks that rely solely on malicious instructions, toxic continuation arises when the model is given a malicious input alongside a partial toxic output, resulting in harmful completions. This vulnerability poses a unique challenge in multimodal settings, where even subtle image variations can disproportionately affect the model's response. To this end, we propose RedDiffuser (RedDiff), the first red teaming framework that uses reinforcement learning to fine-tune diffusion models into generating natural-looking adversarial images that induce toxic continuations. RedDiffuser integrates a greedy search procedure for selecting candidate image prompts with reinforcement fine-tuning that jointly promotes toxic output and semantic coherence. Experiments demonstrate that RedDiffuser significantly increases the toxicity rate in LLaVA outputs by 10.69% and 8.91% on the original and hold-out sets, respectively. It also exhibits strong transferability, increasing toxicity rates on Gemini by 5.1% and on LLaMA-Vision by 26.83%. These findings uncover a cross-modal toxicity amplification vulnerability in current VLM alignment, highlighting the need for robust multimodal red teaming. We will release the RedDiffuser codebase to support future research.

摘要: 视觉语言模型（VLM）容易受到越狱攻击，其中对手绕过安全机制以引出有害输出。在这项工作中，我们研究了这种威胁的一个阴险的变种：有毒的延续。与仅依赖恶意指令的标准越狱不同，当模型被给予恶意输入和部分有毒输出时，就会出现有毒延续，从而导致有害的完成。这种漏洞在多模态环境中构成了独特的挑战，即使是细微的图像变化也会不成比例地影响模型的响应。为此，我们提出了RedDivuser（RedDiff），这是第一个红色团队框架，它使用强化学习来微调扩散模型，以生成诱导有毒延续的自然对抗图像。RedDivuser集成了用于选择候选图像提示的贪婪搜索过程，并进行了强化微调，共同促进了有毒输出和语义一致性。实验表明，RedDiffuser在原始集和保留集上分别显着提高LLaVA输出的毒性率10.69%和8.91%。它还表现出很强的转移性，使Gemini的毒性率提高了5.1%，对LLaMA-Vision的毒性率提高了26.83%。这些发现揭示了当前VLM对齐中的跨模态毒性放大漏洞，突出了对强大的多模态红色团队的需求。我们将发布RedDiffuser代码库以支持未来的研究。



## **41. Impartial Games: A Challenge for Reinforcement Learning**

公正游戏：强化学习的挑战 cs.LG

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2205.12787v5) [paper-pdf](http://arxiv.org/pdf/2205.12787v5)

**Authors**: Bei Zhou, Søren Riis

**Abstract**: AlphaZero-style reinforcement learning (RL) algorithms have achieved superhuman performance in many complex board games such as Chess, Shogi, and Go. However, we showcase that these algorithms encounter significant and fundamental challenges when applied to impartial games, a class where players share game pieces and optimal strategy often relies on abstract mathematical principles. Specifically, we utilize the game of Nim as a concrete and illustrative case study to reveal critical limitations of AlphaZero-style and similar self-play RL algorithms. We introduce a novel conceptual framework distinguishing between champion and expert mastery to evaluate RL agent performance. Our findings reveal that while AlphaZero-style agents can achieve champion-level play on very small Nim boards, their learning progression severely degrades as the board size increases. This difficulty stems not merely from complex data distributions or noisy labels, but from a deeper representational bottleneck: the inherent struggle of generic neural networks to implicitly learn abstract, non-associative functions like parity, which are crucial for optimal play in impartial games. This limitation causes a critical breakdown in the positive feedback loop essential for self-play RL, preventing effective learning beyond rote memorization of frequently observed states. These results align with broader concerns regarding AlphaZero-style algorithms' vulnerability to adversarial attacks, highlighting their inability to truly master all legal game states. Our work underscores that simple hyperparameter adjustments are insufficient to overcome these challenges, establishing a crucial foundation for the development of fundamentally novel algorithmic approaches, potentially involving neuro-symbolic or meta-learning paradigms, to bridge the gap towards true expert-level AI in combinatorial games.

摘要: AlphaZero风格的强化学习（RL）算法在国际象棋、Shogi和Go等许多复杂棋盘游戏中实现了超人的性能。然而，我们展示了这些算法在应用于公正游戏时遇到了重大且根本性的挑战，在公平游戏中，玩家共享游戏棋子，最佳策略通常依赖于抽象的数学原理。具体来说，我们利用Nim游戏作为具体且说明性的案例研究，来揭示AlphaZero风格和类似的自玩RL算法的关键局限性。我们引入了一个新颖的概念框架来区分冠军和专家掌握来评估RL代理的性能。我们的研究结果表明，虽然AlphaZero风格的经纪人可以在非常小的Nim董事会上实现冠军级别的比赛，但随着董事会规模的增加，他们的学习进度会严重下降。这种困难不仅源于复杂的数据分布或嘈杂的标签，还源于更深层次的表示瓶颈：通用神经网络隐性学习抽象、非关联函数（如宇称）的固有斗争，而这些函数对于公正游戏中的最佳玩法至关重要。这种限制导致自玩RL至关重要的正反馈循环严重崩溃，从而阻碍了对频繁观察到的状态死记硬背之外的有效学习。这些结果与人们对AlphaZero式算法容易受到对抗攻击的更广泛担忧一致，凸显了它们无法真正掌握所有合法游戏状态。我们的工作强调，简单的超参数调整不足以克服这些挑战，为开发根本性的新颖算法方法（可能涉及神经符号或元学习范式）奠定了重要基础，以弥合组合游戏中与真正的专家级人工智能的差距。



## **42. Benchmarking Adversarial Patch Selection and Location**

对抗补丁选择和位置基准 cs.CV

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01676v1) [paper-pdf](http://arxiv.org/pdf/2508.01676v1)

**Authors**: Shai Kimhi, Avi Mendlson, Moshe Kimhi

**Abstract**: Adversarial patch attacks threaten the reliability of modern vision models. We present PatchMap, the first spatially exhaustive benchmark of patch placement, built by evaluating over 1.5e8 forward passes on ImageNet validation images. PatchMap reveals systematic hot-spots where small patches (as little as 2% of the image) induce confident misclassifications and large drops in model confidence. To demonstrate its utility, we propose a simple segmentation guided placement heuristic that leverages off the shelf masks to identify vulnerable regions without any gradient queries. Across five architectures-including adversarially trained ResNet50, our method boosts attack success rates by 8 to 13 percentage points compared to random or fixed placements. We publicly release PatchMap and the code implementation. The full PatchMap bench (6.5B predictions, multiple backbones) will be released soon to further accelerate research on location-aware defenses and adaptive attacks.

摘要: 对抗性补丁攻击威胁着现代视觉模型的可靠性。我们介绍了PatchMap，这是第一个空间详尽的补丁放置基准，通过评估ImageNet验证图像超过1.5e8次的正向传递而构建。PatchMap揭示了系统热点，其中小补丁（仅占图像的2%）会导致可信的错误分类和模型置信度大幅下降。为了证明其实用性，我们提出了一种简单的分割引导放置启发式启发式，该启发式利用现成的面具来识别脆弱区域，而无需任何梯度查询。在五种架构中（包括经过对抗训练的ResNet 50），与随机或固定放置相比，我们的方法将攻击成功率提高了8至13个百分点。我们公开发布PatchMap和代码实现。完整的PatchMap平台（6.5B预测，多个主干）将很快发布，以进一步加速对位置感知防御和自适应攻击的研究。



## **43. Practical, Generalizable and Robust Backdoor Attacks on Text-to-Image Diffusion Models**

对文本到图像扩散模型的实用、可推广和鲁棒的后门攻击 cs.CR

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01605v1) [paper-pdf](http://arxiv.org/pdf/2508.01605v1)

**Authors**: Haoran Dai, Jiawen Wang, Ruo Yang, Manali Sharma, Zhonghao Liao, Yuan Hong, Binghui Wang

**Abstract**: Text-to-image diffusion models (T2I DMs) have achieved remarkable success in generating high-quality and diverse images from text prompts, yet recent studies have revealed their vulnerability to backdoor attacks. Existing attack methods suffer from critical limitations: 1) they rely on unnatural adversarial prompts that lack human readability and require massive poisoned data; 2) their effectiveness is typically restricted to specific models, lacking generalizability; and 3) they can be mitigated by recent backdoor defenses.   To overcome these challenges, we propose a novel backdoor attack framework that achieves three key properties: 1) \emph{Practicality}: Our attack requires only a few stealthy backdoor samples to generate arbitrary attacker-chosen target images, as well as ensuring high-quality image generation in benign scenarios. 2) \emph{Generalizability:} The attack is applicable across multiple T2I DMs without requiring model-specific redesign. 3) \emph{Robustness:} The attack remains effective against existing backdoor defenses and adaptive defenses. Our extensive experimental results on multiple T2I DMs demonstrate that with only 10 carefully crafted backdoored samples, our attack method achieves $>$90\% attack success rate with negligible degradation in benign image generation quality. We also conduct human evaluation to validate our attack effectiveness. Furthermore, recent backdoor detection and mitigation methods, as well as adaptive defense tailored to our attack are not sufficiently effective, highlighting the pressing need for more robust defense mechanisms against the proposed attack.

摘要: 文本到图像扩散模型（T2 I DM）在从文本提示生成高质量和多样化的图像方面取得了显着的成功，但最近的研究揭示了它们容易受到后门攻击的影响。现有的攻击方法存在严重局限性：1）它们依赖于非自然的对抗提示，缺乏人类的可读性，并且需要大量有毒数据; 2）它们的有效性通常仅限于特定模型，缺乏概括性; 3）它们可以通过最近的后门防御来缓解。   为了克服这些挑战，我们提出了一种新颖的后门攻击框架，该框架实现了三个关键属性：1）\{实用性}：我们的攻击只需要一些隐秘的后门样本来生成攻击者选择的任意目标图像，并确保在良性场景中生成高质量的图像。2)\{Generalizability：}该攻击适用于多个T2 I DM，无需特定型号的重新设计。3)\{稳健性：}该攻击对现有的后门防御和自适应防御仍然有效。我们对多个T2 I DM的广泛实验结果表明，仅使用10个精心制作的后门样本，我们的攻击方法就可以实现90%的攻击成功率，而良性图像生成质量的下降可以忽略不计。我们还进行人为评估以验证我们的攻击有效性。此外，最近的后门检测和缓解方法以及针对我们的攻击量身定制的自适应防御不够有效，这凸显了针对拟议攻击的更强大防御机制的迫切需要。



## **44. BeDKD: Backdoor Defense based on Dynamic Knowledge Distillation and Directional Mapping Modulator**

BeDKD：基于动态知识蒸馏和方向映射调制器的后门防御 cs.CR

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01595v1) [paper-pdf](http://arxiv.org/pdf/2508.01595v1)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Yinghan Zhou, Changtong dou, Yiming Xue

**Abstract**: Although existing backdoor defenses have gained success in mitigating backdoor attacks, they still face substantial challenges. In particular, most of them rely on large amounts of clean data to weaken the backdoor mapping but generally struggle with residual trigger effects, resulting in persistently high attack success rates (ASR). Therefore, in this paper, we propose a novel Backdoor defense method based on Directional mapping module and adversarial Knowledge Distillation (BeDKD), which balances the trade-off between defense effectiveness and model performance using a small amount of clean and poisoned data. We first introduce a directional mapping module to identify poisoned data, which destroys clean mapping while keeping backdoor mapping on a small set of flipped clean data. Then, the adversarial knowledge distillation is designed to reinforce clean mapping and suppress backdoor mapping through a cycle iteration mechanism between trust and punish distillations using clean and identified poisoned data. We conduct experiments to mitigate mainstream attacks on three datasets, and experimental results demonstrate that BeDKD surpasses the state-of-the-art defenses and reduces the ASR by 98% without significantly reducing the CACC. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD.

摘要: 尽管现有的后门防御措施在缓解后门攻击方面取得了成功，但它们仍然面临着巨大的挑战。特别是，它们中的大多数依赖大量干净的数据来削弱后门映射，但通常会与残余触发效应作斗争，从而导致攻击成功率（ASB）持续很高。因此，本文提出了一种基于方向映射模块和对抗性知识蒸馏（BeDKD）的新型后门防御方法，该方法使用少量干净和有毒数据来平衡防御有效性和模型性能之间的权衡。我们首先引入一个方向性映射模块来识别有毒数据，这会破坏干净映射，同时在一小群翻转干净数据上保留后门映射。然后，对抗性知识蒸馏旨在通过信任和惩罚蒸馏之间的循环迭代机制来加强干净映射并抑制后门映射，使用干净和已识别的有毒数据。我们进行了实验来缓解对三个数据集的主流攻击，实验结果表明，BeDKD超越了最先进的防御能力，并在不显着降低CACC的情况下将ASB降低了98%。我们的代码可在https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD上获取。



## **45. Are All Prompt Components Value-Neutral? Understanding the Heterogeneous Adversarial Robustness of Dissected Prompt in Large Language Models**

所有提示组件都是价值中立的吗？了解大型语言模型中剖析提示的异类对抗鲁棒性 cs.CL

**SubmitDate**: 2025-08-03    [abs](http://arxiv.org/abs/2508.01554v1) [paper-pdf](http://arxiv.org/pdf/2508.01554v1)

**Authors**: Yujia Zheng, Tianhao Li, Haotian Huang, Tianyu Zeng, Jingyu Lu, Chuangxin Chu, Yuekai Huang, Ziyou Jiang, Qian Xiong, Yuyao Ge, Mingyang Li

**Abstract**: Prompt-based adversarial attacks have become an effective means to assess the robustness of large language models (LLMs). However, existing approaches often treat prompts as monolithic text, overlooking their structural heterogeneity-different prompt components contribute unequally to adversarial robustness. Prior works like PromptRobust assume prompts are value-neutral, but our analysis reveals that complex, domain-specific prompts with rich structures have components with differing vulnerabilities. To address this gap, we introduce PromptAnatomy, an automated framework that dissects prompts into functional components and generates diverse, interpretable adversarial examples by selectively perturbing each component using our proposed method, ComPerturb. To ensure linguistic plausibility and mitigate distribution shifts, we further incorporate a perplexity (PPL)-based filtering mechanism. As a complementary resource, we annotate four public instruction-tuning datasets using the PromptAnatomy framework, verified through human review. Extensive experiments across these datasets and five advanced LLMs demonstrate that ComPerturb achieves state-of-the-art attack success rates. Ablation studies validate the complementary benefits of prompt dissection and PPL filtering. Our results underscore the importance of prompt structure awareness and controlled perturbation for reliable adversarial robustness evaluation in LLMs. Code and data are available at https://github.com/Yujiaaaaa/PACP.

摘要: 基于预算的对抗攻击已成为评估大型语言模型（LLM）稳健性的有效手段。然而，现有的方法通常将提示视为单一文本，忽视了它们的结构多样性--不同的提示组件对对抗稳健性的贡献并不平等。Bestrobust等先前的作品假设提示是价值中性的，但我们的分析表明，具有丰富结构的复杂、特定于领域的提示具有不同漏洞的组件。为了解决这一差距，我们引入了EmotAnatomy，这是一个自动化框架，它将提示分解为功能组件，并通过使用我们提出的方法ComPerturb选择性地扰动每个组件来生成多样化的、可解释的对抗性示例。为了确保语言的一致性并减轻分布变化，我们进一步引入了基于困惑度（PPL）的过滤机制。作为补充资源，我们使用AtlantAnatomy框架注释了四个公共描述调整数据集，并通过人类审查进行了验证。这些数据集和五个高级LLM的广泛实验表明，ComPerturb实现了最先进的攻击成功率。消融研究证实了及时解剖和PPL过滤的互补优势。我们的结果强调了即时结构感知和受控扰动对于LLM中可靠的对抗稳健性评估的重要性。代码和数据可在https://github.com/Yujiaaaaa/PACP上获取。



## **46. VWAttacker: A Systematic Security Testing Framework for Voice over WiFi User Equipments**

VWAttacker：WiFi语音用户设备的系统安全测试框架 cs.CR

**SubmitDate**: 2025-08-02    [abs](http://arxiv.org/abs/2508.01469v1) [paper-pdf](http://arxiv.org/pdf/2508.01469v1)

**Authors**: Imtiaz Karim, Hyunwoo Lee, Hassan Asghar, Kazi Samin Mubasshir, Seulgi Han, Mashroor Hasan Bhuiyan, Elisa Bertino

**Abstract**: We present VWAttacker, the first systematic testing framework for analyzing the security of Voice over WiFi (VoWiFi) User Equipment (UE) implementations. VWAttacker includes a complete VoWiFi network testbed that communicates with Commercial-Off-The-Shelf (COTS) UEs based on a simple interface to test the behavior of diverse VoWiFi UE implementations; uses property-guided adversarial testing to uncover security issues in different UEs systematically. To reduce manual effort in extracting and testing properties, we introduce an LLM-based, semi-automatic, and scalable approach for property extraction and testcase (TC) generation. These TCs are systematically mutated by two domain-specific transformations. Furthermore, we introduce two deterministic oracles to detect property violations automatically. Coupled with these techniques, VWAttacker extracts 63 properties from 11 specifications, evaluates 1,116 testcases, and detects 13 issues in 21 UEs. The issues range from enforcing a DH shared secret to 0 to supporting weak algorithms. These issues result in attacks that expose the victim UE's identity or establish weak channels, thus severely hampering the security of cellular networks. We responsibly disclose the findings to all the related vendors. At the time of writing, one of the vulnerabilities has been acknowledged by MediaTek with high severity.

摘要: 我们介绍了VWAttacker，这是第一个用于分析WiFi语音（VoWiFi）用户设备（UE）实施安全性的系统测试框架。VWAttacker包括一个完整的VoWiFi网络测试平台，该测试平台基于简单的接口与商用现货（COTS）UE进行通信，以测试各种VoWiFi UE实施的行为;使用属性引导的对抗测试来系统地揭示不同UE中的安全问题。为了减少提取和测试属性的手动工作，我们引入了一种基于LLM的半自动且可扩展的属性提取和测试用例（TC）生成方法。这些TC通过两种特定于域的转化进行系统性突变。此外，我们还引入了两个确定性先知来自动检测财产违规。结合这些技术，VWAttacker从11个规范中提取63个属性，评估1，116个测试用例，并检测21个UE中的13个问题。这些问题的范围从强制执行DH共享秘密为0到支持弱算法。这些问题会导致暴露受害者UE身份或建立弱通道的攻击，从而严重阻碍蜂窝网络的安全。我们负责任地向所有相关供应商披露调查结果。截至撰写本文时，联发科已承认其中一个严重性漏洞。



## **47. Nakamoto Consensus from Multiple Resources**

来自多方资源的中本共识 cs.CR

Full version of the paper published at AFT25

**SubmitDate**: 2025-08-02    [abs](http://arxiv.org/abs/2508.01448v1) [paper-pdf](http://arxiv.org/pdf/2508.01448v1)

**Authors**: Mirza Ahad Baig, Christoph U. Günther, Krzysztof Pietrzak

**Abstract**: The blocks in the Bitcoin blockchain record the amount of work W that went into creating them through proofs of work. When honest parties control a majority of the work, consensus is achieved by picking the chain with the highest recorded weight. Resources other than work have been considered to secure such longest-chain blockchains. In Chia, blocks record the amount of space S (via a proof of space) and sequential computational steps V (via a VDF).   In this paper, we ask what weight functions {\Gamma}(S,V,W) (that assign a weight to a block as a function of the recorded space, speed, and work) are secure in the sense that whenever the weight of the resources controlled by honest parties is larger than the weight of adversarial parties, the blockchain is secure against private double-spending attacks.   We completely classify such functions in an idealized "continuous" model: {\Gamma}(S,V,W) is secure against private double-spending attacks if and only if it is homogeneous of degree one in the timed resources V and W, i.e., {\alpha}{\Gamma}(S,V,W)={\Gamma}(S,{\alpha}V, {\alpha}W). This includes Bitcoin rule {\Gamma}(S,V,W)=W and Chia rule {\Gamma}(S,V,W) = SV. In a more realistic model where blocks are created at discrete time-points, one additionally needs some mild assumptions on the dependency on S (basically, the weight should not grow too much if S is slightly increased, say linear as in Chia).   Our classification is more general and allows various instantiations of the same resource. It provides a powerful tool for designing new longest-chain blockchains. E.g., consider combining different PoWs to counter centralization, say the Bitcoin PoW W_1 and a memory-hard PoW W_2. Previous work suggested to use W_1+W_2 as weight. Our results show that using {\sqrt}(W_1){\cdot}{\sqrt}(W_2), {\min}{W_1,W_2} are also secure, and we argue that in practice these are much better choices.

摘要: The blocks in the Bitcoin blockchain record the amount of work W that went into creating them through proofs of work. When honest parties control a majority of the work, consensus is achieved by picking the chain with the highest recorded weight. Resources other than work have been considered to secure such longest-chain blockchains. In Chia, blocks record the amount of space S (via a proof of space) and sequential computational steps V (via a VDF).   In this paper, we ask what weight functions {\Gamma}(S,V,W) (that assign a weight to a block as a function of the recorded space, speed, and work) are secure in the sense that whenever the weight of the resources controlled by honest parties is larger than the weight of adversarial parties, the blockchain is secure against private double-spending attacks.   We completely classify such functions in an idealized "continuous" model: {\Gamma}(S,V,W) is secure against private double-spending attacks if and only if it is homogeneous of degree one in the timed resources V and W, i.e., {\alpha}{\Gamma}(S,V,W)={\Gamma}(S,{\alpha}V, {\alpha}W). This includes Bitcoin rule {\Gamma}(S,V,W)=W and Chia rule {\Gamma}(S,V,W) = SV. In a more realistic model where blocks are created at discrete time-points, one additionally needs some mild assumptions on the dependency on S (basically, the weight should not grow too much if S is slightly increased, say linear as in Chia).   Our classification is more general and allows various instantiations of the same resource. It provides a powerful tool for designing new longest-chain blockchains. E.g., consider combining different PoWs to counter centralization, say the Bitcoin PoW W_1 and a memory-hard PoW W_2. Previous work suggested to use W_1+W_2 as weight. Our results show that using {\sqrt}(W_1){\cdot}{\sqrt}(W_2), {\min}{W_1,W_2} are also secure, and we argue that in practice these are much better choices.



## **48. Safety at Scale: A Comprehensive Survey of Large Model and Agent Safety**

大规模安全：大型模型和代理安全的全面调查 cs.CR

706 papers, 60 pages, 3 figures, 14 tables; GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-08-02    [abs](http://arxiv.org/abs/2502.05206v5) [paper-pdf](http://arxiv.org/pdf/2502.05206v5)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Yutao Wu, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Xudong Han, Haonan Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Tim Baldwin, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-powered Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.

摘要: 大型模型在通过大规模预训练进行学习和概括的卓越能力的推动下，迅速发展重塑了人工智能（AI）的格局。这些模型现在是广泛应用的基础，包括对话人工智能、推荐系统、自动驾驶、内容生成、医疗诊断和科学发现。然而，它们的广泛部署也使它们面临巨大的安全风险，引发了对稳健性、可靠性和道德影响的担忧。这项调查对当前大型模型的安全性研究进行了系统性回顾，涵盖视觉基础模型（VFM）、大型语言模型（LLM）、视觉语言预训练（VLP）模型、视觉语言模型（VLM）、扩散模型（DM）和大模型驱动的代理。我们的贡献总结如下：（1）我们对这些模型的安全威胁提出了全面的分类，包括对抗性攻击、数据中毒、后门攻击、越狱和提示注入攻击、能量延迟攻击、数据和模型提取攻击以及新兴的特定于代理的威胁。(2)我们审查为每种类型的攻击提出的防御策略（如果有的话），并总结常用的数据集和安全研究基准。(3)在此基础上，我们确定并讨论了大型模型安全方面的开放挑战，强调全面的安全评估、可扩展且有效的防御机制以及可持续的数据实践的必要性。更重要的是，我们强调研究界集体努力和国际合作的必要性。我们的工作可以为研究人员和从业者提供有用的参考，促进全面防御系统和平台的持续开发，以保护人工智能模型。



## **49. Mitigating Watermark Forgery in Generative Models via Multi-Key Watermarking**

通过多密钥水印缓解生成模型中的水印伪造 cs.CR

**SubmitDate**: 2025-08-02    [abs](http://arxiv.org/abs/2507.07871v2) [paper-pdf](http://arxiv.org/pdf/2507.07871v2)

**Authors**: Toluwani Aremu, Noor Hussein, Munachiso Nwadike, Samuele Poppi, Jie Zhang, Karthik Nandakumar, Neil Gong, Nils Lukas

**Abstract**: Watermarking offers a promising solution for GenAI providers to establish the provenance of their generated content. A watermark is a hidden signal embedded in the generated content, whose presence can later be verified using a secret watermarking key. A security threat to GenAI providers are \emph{forgery attacks}, where malicious users insert the provider's watermark into generated content that was \emph{not} produced by the provider's models, potentially damaging their reputation and undermining trust. One potential defense to resist forgery is using multiple keys to watermark generated content. However, it has been shown that forgery attacks remain successful when adversaries can collect sufficiently many watermarked samples. We propose an improved multi-key watermarking method that resists all surveyed forgery attacks and scales independently of the number of watermarked samples collected by the adversary. Our method accepts content as genuinely watermarked only if \emph{exactly} one watermark is detected. We focus on the image and text modalities, but our detection method is modality-agnostic, since it treats the underlying watermarking method as a black-box. We derive theoretical bounds on forgery-resistance and empirically validate them using Mistral-7B. Our results show a decrease in forgery success from up to $100\%$ using single-key baselines to only $2\%$. While our method resists all surveyed attacks, we find that highly capable, adaptive attackers can still achieve success rates of up to $65\%$ if watermarked content generated using different keys is easily separable.

摘要: 水印为GenAI提供商提供了一个有希望的解决方案，以确定其生成内容的出处。水印是嵌入在生成的内容中的隐藏信号，稍后可以使用秘密水印密钥验证其存在。GenAI提供商面临的安全威胁是\{伪造攻击}，恶意用户将提供商的水印插入到由提供商模型生成的生成内容中，\{not}可能会损害他们的声誉并破坏信任。抵抗伪造的一种潜在防御措施是使用多个密钥来水印生成的内容。然而，事实证明，当对手能够收集足够多的带水印样本时，伪造攻击仍然会成功。我们提出了一种改进的多密钥水印方法，该方法可以抵抗所有调查的伪造攻击，并且独立于对手收集的加水印样本的数量进行扩展。只有当\{exactly}检测到一个水印时，我们的方法才接受内容为真正加了水印。我们专注于图像和文本形态，但我们的检测方法是形态不可知的，因为它将底层水印方法视为黑匣子。我们推导出抗伪造性的理论界限，并使用Mistral-7 B进行经验验证。我们的结果显示，伪造成功率从使用单字基线的高达100美元\%$下降到仅2美元\%$。虽然我们的方法可以抵抗所有调查的攻击，但我们发现，如果使用不同密钥生成的水印内容可以轻松分离，那么高能力的自适应攻击者仍然可以实现高达65美元的成功率。



## **50. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

保护视觉语言模型：缓解基于扰动的攻击中高斯噪音的脆弱性 cs.CV

ICCV 2025

**SubmitDate**: 2025-08-02    [abs](http://arxiv.org/abs/2504.01308v3) [paper-pdf](http://arxiv.org/pdf/2504.01308v3)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yicheng Fu, Yichun Feng, Kin-Man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.

摘要: 视觉语言模型（VLMS）通过合并视觉信息扩展了大型语言模型（LLM）的功能，但它们仍然容易受到越狱攻击，尤其是在处理嘈杂或损坏的图像时。尽管现有的VLM在培训期间采取安全措施来减轻此类攻击，但与噪音增强视觉输入相关的漏洞被忽视了。在这项工作中，我们发现错过噪音增强训练会导致严重的安全漏洞：许多VLM甚至容易受到高斯噪音等简单扰动的影响。为了应对这一挑战，我们提出了Robust-VLGuard，这是一个具有对齐/未对齐图像-文本对的多模式安全数据集，结合了噪音增强微调，可以降低攻击成功率，同时保留VLM的功能。对于更强的基于优化的视觉扰动攻击，我们提出了迪夫Pure-VLM，利用扩散模型将对抗性扰动转换为类高斯噪音，这种噪音可以由具有噪音增强安全微调的VLM进行防御。实验结果表明，扩散模型的分布转移特性与我们微调的VLM很好地一致，显着减轻了不同强度下的对抗扰动。数据集和代码可在https://github.com/JarvisUSTC/DiffPure-RobustVLM上获取。




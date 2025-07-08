# Latest Adversarial Attack Papers
**update at 2025-07-08 10:04:38**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning**

BackFeed：一个高效且标准化的联邦学习后门攻击基准套件 cs.CR

Under review at NeurIPS'25

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04903v1) [paper-pdf](http://arxiv.org/pdf/2507.04903v1)

**Authors**: Thinh Dao, Dung Thuy Nguyen, Khoa D Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) systems are vulnerable to backdoor attacks, where adversaries train their local models on poisoned data and submit poisoned model updates to compromise the global model. Despite numerous proposed attacks and defenses, divergent experimental settings, implementation errors, and unrealistic assumptions hinder fair comparisons and valid conclusions about their effectiveness in real-world scenarios. To address this, we introduce BackFed - a comprehensive benchmark suite designed to standardize, streamline, and reliably evaluate backdoor attacks and defenses in FL, with a focus on practical constraints. Our benchmark offers key advantages through its multi-processing implementation that significantly accelerates experimentation and the modular design that enables seamless integration of new methods via well-defined APIs. With a standardized evaluation pipeline, we envision BackFed as a plug-and-play environment for researchers to comprehensively and reliably evaluate new attacks and defenses. Using BackFed, we conduct large-scale studies of representative backdoor attacks and defenses across both Computer Vision and Natural Language Processing tasks with diverse model architectures and experimental settings. Our experiments critically assess the performance of proposed attacks and defenses, revealing unknown limitations and modes of failures under practical conditions. These empirical insights provide valuable guidance for the development of new methods and for enhancing the security of FL systems. Our framework is openly available at https://github.com/thinh-dao/BackFed.

摘要: 联邦学习（FL）系统很容易受到后门攻击，对手会根据有毒数据训练其本地模型并提交有毒模型更新以损害全局模型。尽管提出了许多攻击和防御，但不同的实验设置、实现错误和不切实际的假设阻碍了公平的比较和关于其在现实世界场景中有效性的有效性的有效结论。为了解决这个问题，我们引入了BackFed --一个全面的基准套件，旨在标准化、简化和可靠地评估FL中的后门攻击和防御，重点关注实际限制。我们的基准测试通过其多处理实施来提供关键优势，可以显着加速实验，并通过定义良好的API实现新方法的无缝集成。通过标准化的评估管道，我们将BackFeed设想为一个即插即用的环境，供研究人员全面可靠地评估新的攻击和防御。使用BackFeed，我们通过不同的模型架构和实验环境对计算机视觉和自然语言处理任务中的代表性后门攻击和防御进行了大规模研究。我们的实验批判性地评估了拟议攻击和防御的性能，揭示了实际条件下未知的限制和失败模式。这些经验见解为新方法的开发和增强FL系统的安全性提供了宝贵的指导。我们的框架可在https://github.com/thinh-dao/BackFed上公开获取。



## **2. Beyond Training-time Poisoning: Component-level and Post-training Backdoors in Deep Reinforcement Learning**

超越训练时中毒：深度强化学习中的学生级和训练后后门 cs.LG

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04883v1) [paper-pdf](http://arxiv.org/pdf/2507.04883v1)

**Authors**: Sanyam Vyas, Alberto Caron, Chris Hicks, Pete Burnap, Vasilios Mavroudis

**Abstract**: Deep Reinforcement Learning (DRL) systems are increasingly used in safety-critical applications, yet their security remains severely underexplored. This work investigates backdoor attacks, which implant hidden triggers that cause malicious actions only when specific inputs appear in the observation space. Existing DRL backdoor research focuses solely on training-time attacks requiring unrealistic access to the training pipeline. In contrast, we reveal critical vulnerabilities across the DRL supply chain where backdoors can be embedded with significantly reduced adversarial privileges. We introduce two novel attacks: (1) TrojanentRL, which exploits component-level flaws to implant a persistent backdoor that survives full model retraining; and (2) InfrectroRL, a post-training backdoor attack which requires no access to training, validation, nor test data. Empirical and analytical evaluations across six Atari environments show our attacks rival state-of-the-art training-time backdoor attacks while operating under much stricter adversarial constraints. We also demonstrate that InfrectroRL further evades two leading DRL backdoor defenses. These findings challenge the current research focus and highlight the urgent need for robust defenses.

摘要: 深度强化学习（DRL）系统越来越多地用于安全关键应用，但其安全性仍然严重不足。这项工作调查了后门攻击，这些攻击植入隐藏触发器，只有当特定输入出现在观察空间中时才会引发恶意操作。现有的DRL后门研究仅关注需要不切实际地访问培训管道的训练时攻击。相比之下，我们揭示了DRL供应链中的关键漏洞，其中后门可以嵌入，对抗特权显着减少。我们引入了两种新颖的攻击：（1）TrojanentRL，它利用组件级缺陷来植入持久的后门，该后门可以在全模型再培训中幸存下来;和（2）InfrectroRL，一种训练后后门攻击，不需要访问训练、验证或测试数据。针对六个Atari环境的经验和分析评估表明，我们的攻击可以与最先进的培训时后门攻击相媲美，同时在更严格的对抗约束下运行。我们还证明InfrectroRL进一步规避了两种主要的DRL后门防御。这些发现挑战了当前的研究重点，并凸显了对强大防御的迫切需要。



## **3. Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection**

基于扩散的对抗性身份操纵用于面部隐私保护 cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2504.21646v2) [paper-pdf](http://arxiv.org/pdf/2504.21646v2)

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The success of face recognition (FR) systems has led to serious privacy concerns due to potential unauthorized surveillance and user tracking on social networks. Existing methods for enhancing privacy fail to generate natural face images that can protect facial privacy. In this paper, we propose diffusion-based adversarial identity manipulation (DiffAIM) to generate natural and highly transferable adversarial faces against malicious FR systems. To be specific, we manipulate facial identity within the low-dimensional latent space of a diffusion model. This involves iteratively injecting gradient-based adversarial identity guidance during the reverse diffusion process, progressively steering the generation toward the desired adversarial faces. The guidance is optimized for identity convergence towards a target while promoting semantic divergence from the source, facilitating effective impersonation while maintaining visual naturalness. We further incorporate structure-preserving regularization to preserve facial structure consistency during manipulation. Extensive experiments on both face verification and identification tasks demonstrate that compared with the state-of-the-art, DiffAIM achieves stronger black-box attack transferability while maintaining superior visual quality. We also demonstrate the effectiveness of the proposed approach for commercial FR APIs, including Face++ and Aliyun.

摘要: 由于社交网络上潜在的未经授权的监视和用户跟踪，面部识别（FR）系统的成功引发了严重的隐私问题。现有的增强隐私的方法无法生成可以保护面部隐私的自然面部图像。在本文中，我们提出了基于扩散的对抗身份操纵（DiffAIM）来生成针对恶意FR系统的自然且高度可转移的对抗面孔。具体来说，我们在扩散模型的低维潜在空间内操纵面部身份。这涉及在反向扩散过程中迭代地注入基于梯度的对抗性身份指导，逐步引导一代人走向所需的对抗性面孔。该指南针对向目标的身份融合进行了优化，同时促进源自源头的语义分歧，促进有效模仿，同时保持视觉自然性。我们进一步结合了结构保留的正规化，以在操作过程中保持面部结构一致性。针对人脸验证和识别任务的大量实验表明，与最新技术相比，迪夫AIM实现了更强的黑匣子攻击可转移性，同时保持了卓越的视觉质量。我们还证明了所提出的方法对商业FR API（包括Face++和Aliyun）的有效性。



## **4. Robustifying 3D Perception through Least-Squares Multi-Agent Graphs Object Tracking**

通过最小平方多智能体图对象跟踪增强3D感知 cs.CV

6 pages, 3 figures, 4 tables

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04762v1) [paper-pdf](http://arxiv.org/pdf/2507.04762v1)

**Authors**: Maria Damanaki, Ioulia Kapsali, Nikos Piperigkos, Alexandros Gkillas, Aris S. Lalos

**Abstract**: The critical perception capabilities of EdgeAI systems, such as autonomous vehicles, are required to be resilient against adversarial threats, by enabling accurate identification and localization of multiple objects in the scene over time, mitigating their impact. Single-agent tracking offers resilience to adversarial attacks but lacks situational awareness, underscoring the need for multi-agent cooperation to enhance context understanding and robustness. This paper proposes a novel mitigation framework on 3D LiDAR scene against adversarial noise by tracking objects based on least-squares graph on multi-agent adversarial bounding boxes. Specifically, we employ the least-squares graph tool to reduce the induced positional error of each detection's centroid utilizing overlapped bounding boxes on a fully connected graph via differential coordinates and anchor points. Hence, the multi-vehicle detections are fused and refined mitigating the adversarial impact, and associated with existing tracks in two stages performing tracking to further suppress the adversarial threat. An extensive evaluation study on the real-world V2V4Real dataset demonstrates that the proposed method significantly outperforms both state-of-the-art single and multi-agent tracking frameworks by up to 23.3% under challenging adversarial conditions, operating as a resilient approach without relying on additional defense mechanisms.

摘要: EdgeAI系统（例如自动驾驶汽车）的关键感知能力需要能够随着时间的推移准确识别和定位场景中的多个对象，从而减轻其影响，从而能够抵御对抗威胁。单代理跟踪提供了对抗攻击的弹性，但缺乏情景感知，这凸显了多代理合作以增强上下文理解和稳健性的必要性。本文提出了一种针对对抗性噪音的新型缓解框架，通过基于多智能体对抗性边界盒上的最小平方图跟踪对象。具体来说，我们使用最小平方图形工具来利用通过差坐标和锚点的完全连接图形上的重叠边界框来减少每个检测的重心的诱导位置误差。因此，多车辆检测被融合和细化，以减轻对抗影响，并在两个阶段与现有轨道相关联，执行跟踪，以进一步抑制对抗威胁。对现实世界V2 V4 Real数据集的广泛评估研究表明，在具有挑战性的对抗条件下，所提出的方法显着比最先进的单代理和多代理跟踪框架高出23.3%，作为一种弹性方法运行，无需依赖额外的防御机制。



## **5. Trojan Horse Prompting: Jailbreaking Conversational Multimodal Models by Forging Assistant Message**

特洛伊木马破解：通过伪造辅助消息破解会话多模态模型 cs.AI

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04673v1) [paper-pdf](http://arxiv.org/pdf/2507.04673v1)

**Authors**: Wei Duan, Li Qian

**Abstract**: The rise of conversational interfaces has greatly enhanced LLM usability by leveraging dialogue history for sophisticated reasoning. However, this reliance introduces an unexplored attack surface. This paper introduces Trojan Horse Prompting, a novel jailbreak technique. Adversaries bypass safety mechanisms by forging the model's own past utterances within the conversational history provided to its API. A malicious payload is injected into a model-attributed message, followed by a benign user prompt to trigger harmful content generation. This vulnerability stems from Asymmetric Safety Alignment: models are extensively trained to refuse harmful user requests but lack comparable skepticism towards their own purported conversational history. This implicit trust in its "past" creates a high-impact vulnerability. Experimental validation on Google's Gemini-2.0-flash-preview-image-generation shows Trojan Horse Prompting achieves a significantly higher Attack Success Rate (ASR) than established user-turn jailbreaking methods. These findings reveal a fundamental flaw in modern conversational AI security, necessitating a paradigm shift from input-level filtering to robust, protocol-level validation of conversational context integrity.

摘要: 对话界面的兴起通过利用对话历史进行复杂推理，极大地增强了LLM的可用性。然而，这种依赖引入了一个未经探索的攻击面。本文介绍了一种新颖的越狱技术特洛伊木马卸载。对手通过在提供给其API的对话历史中伪造模型自己的过去话语来绕过安全机制。恶意有效负载被注入到模型属性消息中，然后是良性用户提示以触发有害内容生成。该漏洞源于不对称安全对齐：模型经过广泛训练以拒绝有害用户请求，但对自己所谓的对话历史缺乏类似的怀疑。这种对其“过去”的隐性信任造成了高影响的脆弱性。对Google Gemini-2.0-flash-preview-image-generation的实验验证表明，特洛伊木马移植的攻击成功率（ASB）比既定的用户越狱方法高得多。这些发现揭示了现代对话人工智能安全性的一个根本缺陷，需要从输入级过滤范式转变为对话上下文完整性的稳健协议级验证。



## **6. Smart Grid: Cyber Attacks, Critical Defense Approaches, and Digital Twin**

智能电网：网络攻击、关键防御方法和数字双胞胎 cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2205.11783v2) [paper-pdf](http://arxiv.org/pdf/2205.11783v2)

**Authors**: Tianming Zheng, Ping Yi, Yue Wu

**Abstract**: As a national critical infrastructure, the smart grid has attracted widespread attention for its cybersecurity issues. The development towards an intelligent, digital, and Internet-connected smart grid has attracted external adversaries for malicious activities. It is necessary to enhance its cybersecurity by both improving the existing defense approaches and introducing novel developed technologies to the smart grid context. As an emerging technology, digital twin (DT) is considered as an enabler for enhanced security. However, the practical implementation is quite challenging. This is due to the knowledge barriers among smart grid designers, security experts, and DT developers. Each single domain is a complicated system covering various components and technologies. As a result, works are needed to sort out relevant contents so that DT can be better embedded in the security architecture design of smart grid.   In order to meet this demand, our paper covers the above three domains, i.e., smart grid, cybersecurity, and DT. Specifically, the paper i) introduces the background of the smart grid; ii) reviews external cyber attacks from attack incidents and attack methods; iii) introduces critical defense approaches in industrial cyber systems, which include device identification, vulnerability discovery, intrusion detection systems (IDSs), honeypots, attribution, and threat intelligence (TI); iv) reviews the relevant content of DT, including its basic concepts, applications in the smart grid, and how DT enhances the security. In the end, the paper puts forward our security considerations on the future development of DT-based smart grid. The survey is expected to help developers break knowledge barriers among smart grid, cybersecurity, and DT, and provide guidelines for future security design of DT-based smart grid.

摘要: 智能电网作为国家关键基础设施，其网络安全问题引起了广泛关注。智能化、数字化和互联网连接的智能电网的发展吸引了外部对手的恶意活动。有必要通过改进现有的防御方法和将新开发的技术引入智能电网环境来增强其网络安全。作为一种新兴技术，数字孪生（DT）被认为是增强安全性的使能者。然而，实际实施相当具有挑战性。这是由于智能电网设计师、安全专家和DT开发人员之间存在知识障碍。每个单一领域都是一个复杂的系统，涵盖各种组件和技术。因此，需要整理相关内容，以便DT更好地嵌入智能电网的安全架构设计中。   为了满足这一需求，我们的论文涵盖了上述三个领域，即智能电网、网络安全和DT。具体来说，本文i）介绍了智能电网的背景; ii）从攻击事件和攻击方法回顾了外部网络攻击; iii）介绍了工业网络系统中的关键防御方法，包括设备识别、漏洞发现、入侵检测系统（IDS）、蜜罐、归因和威胁情报（TI）; iv）回顾DT的相关内容，包括其基本概念、在智能电网中的应用以及DT如何增强安全性。最后，论文提出了基于DT的智能电网未来发展的安全考虑。该调查预计将帮助开发人员打破智能电网、网络安全和DT之间的知识障碍，并为基于DT的智能电网的未来安全设计提供指导。



## **7. Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking**

尾部感知对抗攻击：高效LLM越狱的分布式方法 cs.LG

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04446v1) [paper-pdf](http://arxiv.org/pdf/2507.04446v1)

**Authors**: Tim Beyer, Yan Scholten, Stephan Günnemann, Leo Schwinn

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.

摘要: 为了保证大规模安全、稳健地部署大型语言模型（LLM），准确评估其对抗稳健性至关重要。现有的对抗性攻击通常针对单点贪婪世代的有害响应，忽视了LLM固有的随机性。在本文中，我们提出了一种新颖的对抗稳健性评估框架，该框架对整个输出分布（包括尾部风险）进行显式建模，为模型大规模稳健性提供更好的估计。通过将攻击过程描述为优化和采样之间的资源分配问题，我们确定了计算最优权衡，并表明将采样集成到现有攻击中可将ASB提高高达48%，并将效率提高高达两个数量级。我们的框架还使我们能够分析不同的攻击算法如何影响输出伤害分布。令人惊讶的是，我们发现大多数优化策略对输出危害影响很小。最后，我们引入了一个基于熵最大化的无数据概念验证目标，以演示我们的尾部感知视角如何实现新的优化目标。总体而言，我们的研究结果强调了尾部感知攻击和评估协议对于准确评估和加强LLM安全性的重要性。



## **8. Backdooring Bias ($B^2$) into Stable Diffusion Models**

稳定扩散模型的后门偏差（$B^2$） cs.LG

Accepted to USENIX Security '25

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2406.15213v4) [paper-pdf](http://arxiv.org/pdf/2406.15213v4)

**Authors**: Ali Naseh, Jaechul Roh, Eugene Bagdasarian, Amir Houmansadr

**Abstract**: Recent advances in large text-conditional diffusion models have revolutionized image generation by enabling users to create realistic, high-quality images from textual prompts, significantly enhancing artistic creation and visual communication. However, these advancements also introduce an underexplored attack opportunity: the possibility of inducing biases by an adversary into the generated images for malicious intentions, e.g., to influence public opinion and spread propaganda. In this paper, we study an attack vector that allows an adversary to inject arbitrary bias into a target model. The attack leverages low-cost backdooring techniques using a targeted set of natural textual triggers embedded within a small number of malicious data samples produced with public generative models. An adversary could pick common sequences of words that can then be inadvertently activated by benign users during inference. We investigate the feasibility and challenges of such attacks, demonstrating how modern generative models have made this adversarial process both easier and more adaptable. On the other hand, we explore various aspects of the detectability of such attacks and demonstrate that the model's utility remains intact in the absence of the triggers. Our extensive experiments using over 200,000 generated images and against hundreds of fine-tuned models demonstrate the feasibility of the presented backdoor attack. We illustrate how these biases maintain strong text-image alignment, highlighting the challenges in detecting biased images without knowing that bias in advance. Our cost analysis confirms the low financial barrier (\$10-\$15) to executing such attacks, underscoring the need for robust defensive strategies against such vulnerabilities in diffusion models.

摘要: 大型文本条件扩散模型的最新进展使用户能够根据文本提示创建真实、高质量的图像，从而彻底改变了图像生成，从而显着增强了艺术创作和视觉传达。然而，这些进步也带来了一个未充分探索的攻击机会：对手出于恶意意图将偏见引入生成的图像的可能性，例如，影响舆论并传播宣传。在本文中，我们研究了允许对手将任意偏差注入目标模型的攻击载体。该攻击利用低成本后门技术，使用嵌入公共生成模型生成的少量恶意数据样本中的一组有针对性的自然文本触发器。对手可能会选择常见的单词序列，然后在推理过程中被良性用户无意中激活。我们调查此类攻击的可行性和挑战，展示现代生成模型如何使这种对抗过程变得更容易、更适应。另一方面，我们探索了此类攻击可检测性的各个方面，并证明在没有触发器的情况下，该模型的实用性仍然完好无损。我们使用超过200，000张生成的图像并针对数百个微调模型进行了广泛的实验，证明了所提出的后门攻击的可行性。我们说明了这些偏见如何保持文本与图像的强对齐，强调了在事先不知道偏见的情况下检测偏见图像的挑战。我们的成本分析证实了执行此类攻击的财务障碍较低（10 - 15英镑），强调了针对扩散模型中此类漏洞的强大防御策略的必要性。



## **9. Addressing The Devastating Effects Of Single-Task Data Poisoning In Exemplar-Free Continual Learning**

解决无示例持续学习中单任务数据中毒的破坏性影响 cs.CR

Accepted at CoLLAs 2025

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.04106v1) [paper-pdf](http://arxiv.org/pdf/2507.04106v1)

**Authors**: Stanisław Pawlak, Bartłomiej Twardowski, Tomasz Trzciński, Joost van de Weijer

**Abstract**: Our research addresses the overlooked security concerns related to data poisoning in continual learning (CL). Data poisoning - the intentional manipulation of training data to affect the predictions of machine learning models - was recently shown to be a threat to CL training stability. While existing literature predominantly addresses scenario-dependent attacks, we propose to focus on a more simple and realistic single-task poison (STP) threats. In contrast to previously proposed poisoning settings, in STP adversaries lack knowledge and access to the model, as well as to both previous and future tasks. During an attack, they only have access to the current task within the data stream. Our study demonstrates that even within these stringent conditions, adversaries can compromise model performance using standard image corruptions. We show that STP attacks are able to strongly disrupt the whole continual training process: decreasing both the stability (its performance on past tasks) and plasticity (capacity to adapt to new tasks) of the algorithm. Finally, we propose a high-level defense framework for CL along with a poison task detection method based on task vectors. The code is available at https://github.com/stapaw/STP.git .

摘要: 我们的研究解决了与持续学习（CL）中数据中毒相关的被忽视的安全问题。数据中毒--故意操纵训练数据以影响机器学习模型的预测--最近被证明对CL训练稳定性构成威胁。虽然现有文献主要解决依赖于任务的攻击，但我们建议重点关注更简单、更现实的单任务毒药（STP）威胁。与之前提出的中毒环境相反，STP中的对手缺乏对模型以及之前和未来任务的了解和访问权限。在攻击期间，他们只能访问数据流中的当前任务。我们的研究表明，即使在这些严格的条件下，对手也可以使用标准图像损坏来损害模型性能。我们表明，STP攻击能够强烈破坏整个持续训练过程：降低算法的稳定性（其在过去任务上的性能）和可塑性（适应新任务的能力）。最后，我们提出了一个CL的高级防御框架以及一种基于任务载体的中毒任务检测方法。该代码可在https://github.com/stapaw/STP.git上获取。



## **10. Multichannel Steganography: A Provably Secure Hybrid Steganographic Model for Secure Communication**

多通道隐写术：一种可证明安全的混合隐写术模型，用于安全通信 cs.CR

22 pages, 15 figures, 4 algorithms. This version is a preprint  uploaded to arXiv

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2501.04511v2) [paper-pdf](http://arxiv.org/pdf/2501.04511v2)

**Authors**: Obinna Omego, Michal Bosy

**Abstract**: Secure covert communication in hostile environments requires simultaneously achieving invisibility, provable security guarantees, and robustness against informed adversaries. This paper presents a novel hybrid steganographic framework that unites cover synthesis and cover modification within a unified multichannel protocol. A secret-seeded PRNG drives a lightweight Markov-chain generator to produce contextually plausible cover parameters, which are then masked with the payload and dispersed across independent channels. The masked bit-vector is imperceptibly embedded into conventional media via a variance-aware least-significant-bit algorithm, ensuring that statistical properties remain within natural bounds. We formalize a multichannel adversary model (MC-ATTACK) and prove that, under standard security assumptions, the adversary's distinguishing advantage is negligible, thereby guaranteeing both confidentiality and integrity. Empirical results corroborate these claims: local-variance-guided embedding yields near-lossless extraction (mean BER $<5\times10^{-3}$, correlation $>0.99$) with minimal perceptual distortion (PSNR $\approx100$,dB, SSIM $>0.99$), while key-based masking drives extraction success to zero (BER $\approx0.5$) for a fully informed adversary. Comparative analysis demonstrates that purely distortion-free or invertible schemes fail under the same threat model, underscoring the necessity of hybrid designs. The proposed approach advances high-assurance steganography by delivering an efficient, provably secure covert channel suitable for deployment in high-surveillance networks.

摘要: 敌对环境中的安全秘密通信需要同时实现不可见性、可证明的安全保证以及针对知情对手的稳健性。本文提出了一种新型的混合隐写框架，该框架将覆盖合成和覆盖修改统一到统一的多通道协议中。秘密种子PRNG驱动轻量级的马尔科夫链生成器，以产生上下文上合理的覆盖参数，然后用有效负载掩蔽这些参数并分散在独立的通道中。屏蔽位载体通过方差感知的最低有效位算法以难以察觉的方式嵌入到传统媒体中，确保统计属性保持在自然界限内。我们形式化了多通道对手模型（MC-ATTACK），并证明，在标准安全假设下，对手的区分优势可以忽略不计，从而保证机密性和完整性。经验结果证实了这些说法：局部方差引导嵌入产生了近乎无损的提取（平均BER $<5\times10 &{-3}$，相关性$>0.99$），感知失真最小（PSNR $\approx100 $，DB，SSIM $>0.99$），而对于完全知情的对手，基于密钥的掩蔽将提取成功率推至零（BER $\approx0.5 $）。比较分析表明，纯粹的无失真或可逆的方案在相同的威胁模型下会失败，这凸显了混合设计的必要性。所提出的方法通过提供适合在高监视网络中部署的高效、可证明安全的秘密通道来推进高保证隐写术。



## **11. When There Is No Decoder: Removing Watermarks from Stable Diffusion Models in a No-box Setting**

当没有解码器时：在无框环境中从稳定扩散模型中删除水印 cs.CR

arXiv admin note: text overlap with arXiv:2408.02035

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03646v1) [paper-pdf](http://arxiv.org/pdf/2507.03646v1)

**Authors**: Xiaodong Wu, Tianyi Tang, Xiangman Li, Jianbing Ni, Yong Yu

**Abstract**: Watermarking has emerged as a promising solution to counter harmful or deceptive AI-generated content by embedding hidden identifiers that trace content origins. However, the robustness of current watermarking techniques is still largely unexplored, raising critical questions about their effectiveness against adversarial attacks. To address this gap, we examine the robustness of model-specific watermarking, where watermark embedding is integrated with text-to-image generation in models like latent diffusion models. We introduce three attack strategies: edge prediction-based, box blurring, and fine-tuning-based attacks in a no-box setting, where an attacker does not require access to the ground-truth watermark decoder. Our findings reveal that while model-specific watermarking is resilient against basic evasion attempts, such as edge prediction, it is notably vulnerable to blurring and fine-tuning-based attacks. Our best-performing attack achieves a reduction in watermark detection accuracy to approximately 47.92\%. Additionally, we perform an ablation study on factors like message length, kernel size and decoder depth, identifying critical parameters influencing the fine-tuning attack's success. Finally, we assess several advanced watermarking defenses, finding that even the most robust methods, such as multi-label smoothing, result in watermark extraction accuracy that falls below an acceptable level when subjected to our no-box attacks.

摘要: 水印已成为一种有希望的解决方案，可以通过嵌入跟踪内容起源的隐藏标识符来对抗有害或欺骗性的人工智能生成的内容。然而，当前水印技术的鲁棒性在很大程度上仍未得到探索，这引发了有关其对抗攻击有效性的关键问题。为了解决这一差距，我们研究了特定于模型的水印的鲁棒性，其中水印嵌入与潜在扩散模型等模型中的文本到图像生成相集成。我们引入了三种攻击策略：基于边缘预测的攻击、框模糊攻击和无框设置中的基于微调的攻击，其中攻击者不需要访问地面真相水印解码器。我们的研究结果表明，虽然特定于模型的水印可以抵御边缘预测等基本规避尝试，但它特别容易受到模糊和基于微调的攻击。我们性能最好的攻击将水印检测准确率降低至约47.92%。此外，我们还对消息长度、内核大小和解码器深度等因素进行了消融研究，确定影响微调攻击成功的关键参数。最后，我们评估了几种先进的水印防御，发现即使是最稳健的方法（例如多标签平滑），在受到无框攻击时，水印提取准确性也会低于可接受的水平。



## **12. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

探索LLM中的潜在子空间以实现人工智能安全：识别和操纵敌对状态 cs.LG

4 figures

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2503.09066v2) [paper-pdf](http://arxiv.org/pdf/2503.09066v2)

**Authors**: Xin Wei Chia, Swee Liang Wong, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.

摘要: 大型语言模型（LLM）在各种任务中表现出了非凡的能力，但它们仍然容易受到对抗操纵的影响，例如通过提示注入攻击进行越狱。这些攻击绕过安全机制来生成受限制或有害内容。在这项研究中，我们通过从LLM中提取隐藏激活来研究安全和越狱状态的潜在子空间。受神经科学中吸引子动力学的启发，我们假设LLM激活会进入半稳定状态，可以识别和扰动这些状态以引发状态转变。使用降维技术，我们预测安全和越狱反应的激活，以揭示低维空间中的潜在子空间。然后，我们推导出一个扰动载体，当将其应用于安全表示时，会将模型转向越狱状态。我们的结果表明，这种因果干预会在提示子集中导致具有统计学意义的越狱反应。接下来，我们探讨了这些扰动如何在模型的层中传播，测试诱导的状态变化是保持局部化还是在整个网络中级联。我们的研究结果表明，有针对性的扰动会导致激活和模型响应的明显变化。我们的方法为潜在的主动防御铺平了道路，从传统的基于护栏的方法转向先发制人的、模型不可知的技术，可以在表示层面中和对抗状态。



## **13. On the Limits of Robust Control Under Adversarial Disturbances**

论对抗扰动下鲁棒控制的极限 eess.SY

Extended version of a manuscript submitted to IEEE Transactions on  Automatic Control, July 2025

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03630v1) [paper-pdf](http://arxiv.org/pdf/2507.03630v1)

**Authors**: Paul Trodden, José M. Maestre, Hideaki Ishii

**Abstract**: This paper addresses a fundamental and important question in control: under what conditions does there fail to exist a robust control policy that keeps the state of a constrained linear system within a target set, despite bounded disturbances? This question has practical implications for actuator and sensor specification, feasibility analysis for reference tracking, and the design of adversarial attacks in cyber-physical systems. While prior research has predominantly focused on using optimization to compute control-invariant sets to ensure feasible operation, our work complements these approaches by characterizing explicit sufficient conditions under which robust control is fundamentally infeasible. Specifically, we derive novel closed-form, algebraic expressions that relate the size of a disturbance set -- modelled as a scaled version of a basic shape -- to the system's spectral properties and the geometry of the constraint sets.

摘要: 本文解决了控制中的一个基本而重要的问题：在什么条件下，不存在鲁棒控制策略，即使存在有界干扰，也不存在将受约束线性系统的状态保持在目标集中？这个问题对致动器和传感器规范、参考跟踪的可行性分析以及网络物理系统中对抗攻击的设计具有实际影响。虽然之前的研究主要集中在使用优化来计算控制不变集以确保可行的操作，但我们的工作通过描述鲁棒控制从根本上不可行的显式充分条件来补充这些方法。具体来说，我们推导出新颖的封闭形式的代数表达，将干扰集的大小（建模为基本形状的缩放版本）与系统的谱属性和约束集的几何形状联系起来。



## **14. Beyond Weaponization: NLP Security for Medium and Lower-Resourced Languages in Their Own Right**

超越再殖民化：中等和低资源语言的NLP安全性本身 cs.CL

Pre-print

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03473v1) [paper-pdf](http://arxiv.org/pdf/2507.03473v1)

**Authors**: Heather Lent

**Abstract**: Despite mounting evidence that multilinguality can be easily weaponized against language models (LMs), works across NLP Security remain overwhelmingly English-centric. In terms of securing LMs, the NLP norm of "English first" collides with standard procedure in cybersecurity, whereby practitioners are expected to anticipate and prepare for worst-case outcomes. To mitigate worst-case outcomes in NLP Security, researchers must be willing to engage with the weakest links in LM security: lower-resourced languages. Accordingly, this work examines the security of LMs for lower- and medium-resourced languages. We extend existing adversarial attacks for up to 70 languages to evaluate the security of monolingual and multilingual LMs for these languages. Through our analysis, we find that monolingual models are often too small in total number of parameters to ensure sound security, and that while multilinguality is helpful, it does not always guarantee improved security either. Ultimately, these findings highlight important considerations for more secure deployment of LMs, for communities of lower-resourced languages.

摘要: 尽管越来越多的证据表明多语言可以很容易地被武器化来对抗语言模型（LM），但NLP Security的工作仍然绝大多数以英语为中心。在确保LM方面，“英语优先”的NLP规范与网络安全中的标准程序相冲突，而从业者需要预测并为最坏情况的结果做好准备。为了减轻NLP安全中最坏的情况，研究人员必须愿意接触LM安全中最薄弱的环节：资源较少的语言。因此，这项工作考察了低年级和中等资源语言的LM的安全性。我们将现有的对抗性攻击扩展到多达70种语言，以评估这些语言的单语和多语言LM的安全性。通过我们的分析，我们发现单语模型的参数总数通常太小，无法确保良好的安全性，而且虽然多语言很有帮助，但它也不总是保证安全性的提高。最终，这些发现强调了为资源较少的语言社区更安全地部署LM的重要考虑因素。



## **15. Evaluating the Evaluators: Trust in Adversarial Robustness Tests**

评估评估者：对对抗稳健性测试的信任 cs.CR

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03450v1) [paper-pdf](http://arxiv.org/pdf/2507.03450v1)

**Authors**: Antonio Emanuele Cinà, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Despite significant progress in designing powerful adversarial evasion attacks for robustness verification, the evaluation of these methods often remains inconsistent and unreliable. Many assessments rely on mismatched models, unverified implementations, and uneven computational budgets, which can lead to biased results and a false sense of security. Consequently, robustness claims built on such flawed testing protocols may be misleading and give a false sense of security. As a concrete step toward improving evaluation reliability, we present AttackBench, a benchmark framework developed to assess the effectiveness of gradient-based attacks under standardized and reproducible conditions. AttackBench serves as an evaluation tool that ranks existing attack implementations based on a novel optimality metric, which enables researchers and practitioners to identify the most reliable and effective attack for use in subsequent robustness evaluations. The framework enforces consistent testing conditions and enables continuous updates, making it a reliable foundation for robustness verification.

摘要: 尽管在设计强大的对抗规避攻击以进行稳健性验证方面取得了重大进展，但这些方法的评估往往仍然不一致且不可靠。许多评估依赖于不匹配的模型、未经验证的实现和不均衡的计算预算，这可能会导致有偏见的结果和错误的安全感。因此，建立在此类有缺陷的测试协议上的稳健性声明可能会具有误导性，并给人一种错误的安全感。作为提高评估可靠性的具体步骤，我们提出了AttackBench，这是一个基准框架，旨在评估标准化和可重复条件下基于梯度的攻击的有效性。AttackBench作为一种评估工具，根据新颖的最优性指标对现有攻击实施进行排名，使研究人员和从业者能够识别最可靠、最有效的攻击，以用于后续的稳健性评估。该框架强制执行一致的测试条件并实现持续更新，使其成为稳健性验证的可靠基础。



## **16. Rectifying Adversarial Sample with Low Entropy Prior for Test-Time Defense**

用低熵先验纠正对抗样本以进行测试时防御 cs.CV

To appear in IEEEE Transactions on Multimedia

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03427v1) [paper-pdf](http://arxiv.org/pdf/2507.03427v1)

**Authors**: Lina Ma, Xiaowei Fu, Fuxiang Huang, Xinbo Gao, Lei Zhang

**Abstract**: Existing defense methods fail to defend against unknown attacks and thus raise generalization issue of adversarial robustness. To remedy this problem, we attempt to delve into some underlying common characteristics among various attacks for generality. In this work, we reveal the commonly overlooked low entropy prior (LE) implied in various adversarial samples, and shed light on the universal robustness against unseen attacks in inference phase. LE prior is elaborated as two properties across various attacks as shown in Fig. 1 and Fig. 2: 1) low entropy misclassification for adversarial samples and 2) lower entropy prediction for higher attack intensity. This phenomenon stands in stark contrast to the naturally distributed samples. The LE prior can instruct existing test-time defense methods, thus we propose a two-stage REAL approach: Rectify Adversarial sample based on LE prior for test-time adversarial rectification. Specifically, to align adversarial samples more closely with clean samples, we propose to first rectify adversarial samples misclassified with low entropy by reverse maximizing prediction entropy, thereby eliminating their adversarial nature. To ensure the rectified samples can be correctly classified with low entropy, we carry out secondary rectification by forward minimizing prediction entropy, thus creating a Max-Min entropy optimization scheme. Further, based on the second property, we propose an attack-aware weighting mechanism to adaptively adjust the strengths of Max-Min entropy objectives. Experiments on several datasets show that REAL can greatly improve the performance of existing sample rectification models.

摘要: 现有的防御方法未能防御未知攻击，从而提出了对抗鲁棒性的一般化问题。为了解决这个问题，我们试图深入研究各种攻击之间的一些潜在共同特征，以获取一般性。在这项工作中，我们揭示了各种对抗样本中隐含的普遍被忽视的低熵先验（LE），并揭示了在推理阶段针对不可见攻击的普遍鲁棒性。LE先验被描述为各种攻击的两个属性，如图1和图2所示：1）对抗性样本的低熵误分类; 2）针对更高的攻击强度的较低熵预测。这种现象与自然分布的样本形成鲜明对比。LE先验可以指导现有的测试时防御方法，因此我们提出了一种两阶段REAL方法：基于LE先验来纠正对抗样本，用于测试时对抗纠正。具体来说，为了更接近地将对抗性样本与干净样本对齐，我们建议首先通过反向最大化预测熵来纠正错误分类的对抗性样本，从而消除它们的对抗性本质。为了确保纠正后的样本能够以低信息正确分类，我们通过向前最小化预测信息进行二次纠正，从而创建了Max-Min信息优化方案。此外，基于第二个属性，我们提出了一个攻击感知加权机制，自适应调整的最大-最小熵目标的强度。在多个数据集上的实验表明，REAL可以大大提高现有样本校正模型的性能。



## **17. Breaking the Bulkhead: Demystifying Cross-Namespace Reference Vulnerabilities in Kubernetes Operators**

打破障碍：揭开Kubernetes运营商中的跨空间引用漏洞的神秘面纱 cs.CR

12 pages

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03387v1) [paper-pdf](http://arxiv.org/pdf/2507.03387v1)

**Authors**: Andong Chen, Zhaoxuan Jin, Ziyi Guo, Yan Chen

**Abstract**: Kubernetes Operators, automated tools designed to manage application lifecycles within Kubernetes clusters, extend the functionalities of Kubernetes, and reduce the operational burden on human engineers. While Operators significantly simplify DevOps workflows, they introduce new security risks. In particular, Kubernetes enforces namespace isolation to separate workloads and limit user access, ensuring that users can only interact with resources within their authorized namespaces. However, Kubernetes Operators often demand elevated privileges and may interact with resources across multiple namespaces. This introduces a new class of vulnerabilities, the Cross-Namespace Reference Vulnerability. The root cause lies in the mismatch between the declared scope of resources and the implemented scope of the Operator logic, resulting in Kubernetes being unable to properly isolate the namespace. Leveraging such vulnerability, an adversary with limited access to a single authorized namespace may exploit the Operator to perform operations affecting other unauthorized namespaces, causing Privilege Escalation and further impacts. To the best of our knowledge, this paper is the first to systematically investigate the security vulnerability of Kubernetes Operators. We present Cross-Namespace Reference Vulnerability with two strategies, demonstrating how an attacker can bypass namespace isolation. Through large-scale measurements, we found that over 14% of Operators in the wild are potentially vulnerable. Our findings have been reported to the relevant developers, resulting in 7 confirmations and 6 CVEs by the time of submission, affecting vendors including ****** and ******, highlighting the critical need for enhanced security practices in Kubernetes Operators. To mitigate it, we also open-source the static analysis suite to benefit the ecosystem.

摘要: Kubernetes Operators是一种自动化工具，旨在管理Kubernetes集群内的应用程序生命周期，扩展Kubernetes的功能，并减轻人类工程师的运营负担。虽然运营商显着简化了DevOps工作流程，但他们也引入了新的安全风险。特别是，Kubernetes强制执行命名空间隔离以分离工作负载并限制用户访问，确保用户只能与其授权命名空间内的资源交互。然而，Kubernetes操作员通常需要更高的特权，并且可能会与跨多个名称空间的资源交互。这引入了一类新的漏洞，即跨空间引用漏洞。根本原因在于声明的资源范围与Operator逻辑的实现范围不匹配，导致Kubernetes无法正确隔离命名空间。利用此类漏洞，对单个授权命名空间的访问权限有限的对手可能会利用运营商执行影响其他未经授权命名空间的操作，从而导致特权升级和进一步影响。据我们所知，本文是第一篇系统性研究Kubernetes Operators安全漏洞的论文。我们通过两种策略展示跨命名空间引用漏洞，展示攻击者如何绕过命名空间隔离。通过大规模测量，我们发现超过14%的野外经营者存在潜在的脆弱性。我们的调查结果已报告给相关开发人员，截至提交时已获得7项确认和6项CVE，影响了 * 和 * 等供应商，凸显了Kubernetes Operators对增强安全实践的迫切需求。为了缓解这种情况，我们还开源了静态分析套件，以造福生态系统。



## **18. Fault Sneaking Attack: a Stealthy Framework for Misleading Deep Neural Networks**

故障潜行攻击：误导深度神经网络的隐形框架 cs.LG

Accepted by the 56th Design Automation Conference (DAC 2019)

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/1905.12032v2) [paper-pdf](http://arxiv.org/pdf/1905.12032v2)

**Authors**: Pu Zhao, Siyue Wang, Cheng Gongye, Yanzhi Wang, Yunsi Fei, Xue Lin

**Abstract**: Despite the great achievements of deep neural networks (DNNs), the vulnerability of state-of-the-art DNNs raises security concerns of DNNs in many application domains requiring high reliability.We propose the fault sneaking attack on DNNs, where the adversary aims to misclassify certain input images into any target labels by modifying the DNN parameters. We apply ADMM (alternating direction method of multipliers) for solving the optimization problem of the fault sneaking attack with two constraints: 1) the classification of the other images should be unchanged and 2) the parameter modifications should be minimized. Specifically, the first constraint requires us not only to inject designated faults (misclassifications), but also to hide the faults for stealthy or sneaking considerations by maintaining model accuracy. The second constraint requires us to minimize the parameter modifications (using L0 norm to measure the number of modifications and L2 norm to measure the magnitude of modifications). Comprehensive experimental evaluation demonstrates that the proposed framework can inject multiple sneaking faults without losing the overall test accuracy performance.

摘要: 尽管深度神经网络（DNN）取得了巨大成就，但最先进的DNN的脆弱性在许多要求高可靠性的应用领域中引发了DNN的安全担忧。我们提出了对DNN的故障潜行攻击，对手的目标是通过修改DNN参数将某些输入图像错误分类到任何目标标签中。我们应用ADMM（交替方向乘数法）来解决故障潜行攻击的优化问题，具有两个约束：1）其他图像的分类应该保持不变; 2）参数修改应该最小化。具体来说，第一个约束要求我们不仅注入指定的故障（错误分类），还要求我们通过保持模型准确性来隐藏故障，以进行隐蔽或隐蔽的考虑。第二个约束要求我们最小化参数修改（使用L0规范来衡量修改的数量，使用L2规范来衡量修改的幅度）。综合实验评估表明，所提出的框架可以在不损失整体测试准确性性能的情况下注入多个隐蔽故障。



## **19. On the Adversarial Robustness of Graph Neural Networks with Graph Reduction**

基于图约简的图神经网络的对抗鲁棒性 cs.LG

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2412.05883v2) [paper-pdf](http://arxiv.org/pdf/2412.05883v2)

**Authors**: Kerui Wu, Ka-Ho Chow, Wenqi Wei, Lei Yu

**Abstract**: As Graph Neural Networks (GNNs) become increasingly popular for learning from large-scale graph data across various domains, their susceptibility to adversarial attacks when using graph reduction techniques for scalability remains underexplored. In this paper, we present an extensive empirical study to investigate the impact of graph reduction techniques, specifically graph coarsening and sparsification, on the robustness of GNNs against adversarial attacks. Through extensive experiments involving multiple datasets and GNN architectures, we examine the effects of four sparsification and six coarsening methods on the poisoning attacks. Our results indicate that, while graph sparsification can mitigate the effectiveness of certain poisoning attacks, such as Mettack, it has limited impact on others, like PGD. Conversely, graph coarsening tends to amplify the adversarial impact, significantly reducing classification accuracy as the reduction ratio decreases. Additionally, we provide a novel analysis of the causes driving these effects and examine how defensive GNN models perform under graph reduction, offering practical insights for designing robust GNNs within graph acceleration systems.

摘要: 随着图神经网络（GNN）在从各个领域的大规模图数据中学习方面变得越来越受欢迎，但在使用图约简技术来实现可扩展性时，它们对对抗攻击的敏感性仍然没有得到充分的研究。在本文中，我们进行了一项广泛的实证研究，以调查图约简技术（特别是图粗化和稀疏化）对GNN对抗攻击的鲁棒性的影响。通过涉及多个数据集和GNN架构的广泛实验，我们研究了四种稀疏化和六种粗化方法对中毒攻击的影响。我们的结果表明，虽然图稀疏化可以降低某些中毒攻击（例如Mettack）的有效性，但它对其他中毒攻击（例如PVD）的影响有限。相反，图形粗化往往会放大对抗影响，随着缩减率的降低，分类准确性会显着降低。此外，我们还对驱动这些效应的原因进行了新颖的分析，并研究了防御性GNN模型在图约简下的表现，为在图加速系统中设计稳健的GNN提供了实用的见解。



## **20. Adopting a human developmental visual diet yields robust, shape-based AI vision**

采用人类发育视觉饮食产生强大的、基于形状的人工智能视觉 cs.LG

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.03168v1) [paper-pdf](http://arxiv.org/pdf/2507.03168v1)

**Authors**: Zejin Lu, Sushrut Thorat, Radoslaw M Cichy, Tim C Kietzmann

**Abstract**: Despite years of research and the dramatic scaling of artificial intelligence (AI) systems, a striking misalignment between artificial and human vision persists. Contrary to humans, AI heavily relies on texture-features rather than shape information, lacks robustness to image distortions, remains highly vulnerable to adversarial attacks, and struggles to recognise simple abstract shapes within complex backgrounds. To close this gap, we here introduce a solution that arises from a previously underexplored direction: rather than scaling up, we take inspiration from how human vision develops from early infancy into adulthood. We quantified the visual maturation by synthesising decades of psychophysical and neurophysiological research into a novel developmental visual diet (DVD) for AI vision. We show that guiding AI systems through this human-inspired curriculum produces models that closely align with human behaviour on every hallmark of robust vision tested yielding the strongest reported reliance on shape information to date, abstract shape recognition beyond the state of the art, higher robustness to image corruptions, and stronger resilience to adversarial attacks. By outperforming high parameter AI foundation models trained on orders of magnitude more data, we provide evidence that robust AI vision can be achieved by guiding the way how a model learns, not merely how much it learns, offering a resource-efficient route toward safer and more human-like artificial visual systems.

摘要: 尽管经过多年的研究和人工智能（AI）系统的巨大规模，但人工视觉和人类视觉之间仍然存在明显的不一致。与人类相反，人工智能严重依赖纹理特征而不是形状信息，缺乏对图像失真的鲁棒性，仍然极易受到对抗攻击，并且很难识别复杂背景中的简单抽象形状。为了缩小这一差距，我们在这里引入了一种源自之前未充分探索的方向的解决方案：我们没有扩大规模，而是从人类视觉从婴儿早期到成年期的发展中汲取灵感。我们通过将数十年的心理物理和神经生理研究综合为人工智能视觉的新型发育视觉饮食（DVD）来量化视觉成熟。我们表明，引导人工智能系统完成这一受人类启发的课程，可以在测试的稳健视觉的每个标志上产生与人类行为密切一致的模型，从而产生迄今为止对形状信息的最强依赖性、超越最先进水平的抽象形状识别、对图像损坏更高的鲁棒性以及对对抗性攻击更强的韧性。通过优于在数量级更多数据上训练的高参数人工智能基础模型，我们提供了证据，表明可以通过指导模型学习的方式（而不仅仅是学习多少）来实现强大的人工智能愿景，从而提供了一条资源高效的途径，以实现更安全、更像人的人工视觉系统。



## **21. Adversarial Manipulation of Reasoning Models using Internal Representations**

使用内部表示的推理模型的对抗性操纵 cs.CL

Accepted to the ICML 2025 Workshop on Reliable and Responsible  Foundation Models (R2FM). 20 pages, 12 figures

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.03167v1) [paper-pdf](http://arxiv.org/pdf/2507.03167v1)

**Authors**: Kureha Yamaguchi, Benjamin Etheridge, Andy Arditi

**Abstract**: Reasoning models generate chain-of-thought (CoT) tokens before their final output, but how this affects their vulnerability to jailbreak attacks remains unclear. While traditional language models make refusal decisions at the prompt-response boundary, we find evidence that DeepSeek-R1-Distill-Llama-8B makes these decisions within its CoT generation. We identify a linear direction in activation space during CoT token generation that predicts whether the model will refuse or comply -- termed the "caution" direction because it corresponds to cautious reasoning patterns in the generated text. Ablating this direction from model activations increases harmful compliance, effectively jailbreaking the model. We additionally show that intervening only on CoT token activations suffices to control final outputs, and that incorporating this direction into prompt-based attacks improves success rates. Our findings suggest that the chain-of-thought itself is a promising new target for adversarial manipulation in reasoning models.   Code available at https://github.com/ky295/reasoning-manipulation

摘要: 推理模型在最终输出之前生成思想链（CoT）代币，但这如何影响其对越狱攻击的脆弱性尚不清楚。虽然传统语言模型在预算-响应边界做出拒绝决策，但我们发现证据表明DeepSeek-R1-Distill-Llama-8B在其CoT一代内做出这些决策。我们在CoT令牌生成过程中识别激活空间中的线性方向，该方向预测模型是否会拒绝或遵守--称为“谨慎”方向，因为它对应于生成的文本中的谨慎推理模式。从模型激活中汲取这个方向会增加有害的合规性，实际上是对模型的越狱。我们还表明，仅干预CoT代币激活就足以控制最终输出，并且将此方向纳入基于预算的攻击可以提高成功率。我们的研究结果表明，思想链本身是推理模型中对抗操纵的一个有希望的新目标。   代码可访问https://github.com/ky295/reasoning-manipulation



## **22. LIAR: Leveraging Inference Time Alignment (Best-of-N) to Jailbreak LLMs in Seconds**

LIAR：利用推理时间对齐（最佳N）在几秒钟内越狱LLM cs.CL

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2412.05232v3) [paper-pdf](http://arxiv.org/pdf/2412.05232v3)

**Authors**: James Beetham, Souradip Chakraborty, Mengdi Wang, Furong Huang, Amrit Singh Bedi, Mubarak Shah

**Abstract**: Jailbreak attacks expose vulnerabilities in safety-aligned LLMs by eliciting harmful outputs through carefully crafted prompts. Existing methods rely on discrete optimization or trained adversarial generators, but are slow, compute-intensive, and often impractical. We argue that these inefficiencies stem from a mischaracterization of the problem. Instead, we frame jailbreaks as inference-time misalignment and introduce LIAR (Leveraging Inference-time misAlignment to jailbReak), a fast, black-box, best-of-$N$ sampling attack requiring no training. LIAR matches state-of-the-art success rates while reducing perplexity by $10\times$ and Time-to-Attack from hours to seconds. We also introduce a theoretical "safety net against jailbreaks" metric to quantify safety alignment strength and derive suboptimality bounds. Our work offers a simple yet effective tool for evaluating LLM robustness and advancing alignment research.

摘要: 越狱攻击通过精心设计的提示引发有害输出，暴露了安全一致的LLM中的漏洞。现有的方法依赖于离散优化或经过训练的对抗生成器，但速度缓慢、计算密集型，并且通常不切实际。我们认为，这些低效率源于对问题的错误描述。相反，我们将越狱定义为推断时间错位，并引入LIAR（利用推断时间错位到jailbReak），这是一种快速、黑匣子、最佳N $抽样攻击，无需训练。LIAR与最先进的成功率相匹配，同时将困惑度减少10美元乘以$，攻击时间从数小时缩短到数秒。我们还引入了理论上的“防止越狱的安全网”指标来量化安全对齐强度并推导出次优界限。我们的工作为评估LLM稳健性和推进对齐研究提供了一个简单而有效的工具。



## **23. LoRA as a Flexible Framework for Securing Large Vision Systems**

LoRA作为保护大型视觉系统的灵活框架 cs.CV

Updated pre-print. Under review

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2506.00661v2) [paper-pdf](http://arxiv.org/pdf/2506.00661v2)

**Authors**: Zander W. Blasingame, Richard E. Neddo, Chen Liu

**Abstract**: Adversarial attacks have emerged as a critical threat to autonomous driving systems. These attacks exploit the underlying neural network, allowing small -- nearly invisible -- perturbations to completely alter the behavior of such systems in potentially malicious ways. E.g., causing a traffic sign classification network to misclassify a stop sign as a speed limit sign. Prior working in hardening such systems to adversarial attacks have looked at robust training of the system or adding additional pre-processing steps to the input pipeline. Such solutions either have a hard time generalizing, require knowledge of the adversarial attacks during training, or are computationally undesirable. Instead, we propose to take insights for parameter efficient fine-tuning and use low-rank adaptation (LoRA) to train a lightweight security patch -- enabling us to dynamically patch a large preexisting vision system as new vulnerabilities are discovered. We demonstrate that our framework can patch a pre-trained model to improve classification accuracy by up to 78.01% in the presence of adversarial examples.

摘要: 对抗性攻击已成为自动驾驶系统的严重威胁。这些攻击利用底层神经网络，允许微小的（几乎不可见的）扰动以潜在恶意的方式完全改变此类系统的行为。例如，导致交通标志分类网络将停车标志错误分类为限速标志。之前在加强此类系统抵御对抗攻击方面的工作已经考虑了系统的稳健训练或向输入管道添加额外的预处理步骤。此类解决方案要么很难概括，需要了解训练期间的对抗攻击，要么在计算上不理想。相反，我们建议深入了解参数高效微调，并使用低等级自适应（LoRA）来训练轻量级安全补丁--使我们能够在发现新漏洞时动态修补大型先前存在的视觉系统。我们证明，我们的框架可以修补预训练的模型，在存在对抗性示例的情况下将分类准确率提高高达78.01%。



## **24. Is Reasoning All You Need? Probing Bias in the Age of Reasoning Language Models**

你需要的就是推理吗？探索推理语言模型时代的偏见 cs.CL

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02799v1) [paper-pdf](http://arxiv.org/pdf/2507.02799v1)

**Authors**: Riccardo Cantini, Nicola Gabriele, Alessio Orsino, Domenico Talia

**Abstract**: Reasoning Language Models (RLMs) have gained traction for their ability to perform complex, multi-step reasoning tasks through mechanisms such as Chain-of-Thought (CoT) prompting or fine-tuned reasoning traces. While these capabilities promise improved reliability, their impact on robustness to social biases remains unclear. In this work, we leverage the CLEAR-Bias benchmark, originally designed for Large Language Models (LLMs), to investigate the adversarial robustness of RLMs to bias elicitation. We systematically evaluate state-of-the-art RLMs across diverse sociocultural dimensions, using an LLM-as-a-judge approach for automated safety scoring and leveraging jailbreak techniques to assess the strength of built-in safety mechanisms. Our evaluation addresses three key questions: (i) how the introduction of reasoning capabilities affects model fairness and robustness; (ii) whether models fine-tuned for reasoning exhibit greater safety than those relying on CoT prompting at inference time; and (iii) how the success rate of jailbreak attacks targeting bias elicitation varies with the reasoning mechanisms employed. Our findings reveal a nuanced relationship between reasoning capabilities and bias safety. Surprisingly, models with explicit reasoning, whether via CoT prompting or fine-tuned reasoning traces, are generally more vulnerable to bias elicitation than base models without such mechanisms, suggesting reasoning may unintentionally open new pathways for stereotype reinforcement. Reasoning-enabled models appear somewhat safer than those relying on CoT prompting, which are particularly prone to contextual reframing attacks through storytelling prompts, fictional personas, or reward-shaped instructions. These results challenge the assumption that reasoning inherently improves robustness and underscore the need for more bias-aware approaches to reasoning design.

摘要: 推理语言模型（RLM）通过诸如思想链（CoT）提示或微调推理轨迹等机制来执行复杂的多步推理任务的能力已经获得了关注。虽然这些功能有望提高可靠性，但它们对社会偏见鲁棒性的影响仍不清楚。在这项工作中，我们利用CLEAR-Bias基准，最初是为大型语言模型（LLM）设计的，来研究RLM对偏见启发的对抗鲁棒性。我们在不同的社会文化维度上系统地评估最先进的RLM，使用LLM作为自动安全评分的评判方法，并利用越狱技术来评估内置安全机制的强度。我们的评估解决了三个关键问题：（i）推理能力的引入如何影响模型的公平性和稳健性;（ii）为推理进行微调的模型是否比在推理时依赖CoT提示的模型表现出更大的安全性;（iii）针对偏见引发的越狱攻击的成功率如何随着所采用的推理机制而变化。我们的研究结果揭示了推理能力和偏见安全性之间的微妙关系。令人惊讶的是，具有显式推理的模型，无论是通过CoT提示还是微调推理痕迹，通常比没有此类机制的基本模型更容易受到偏见引发，这表明推理可能会无意中为刻板印象强化开辟新的途径。支持推理的模型似乎比依赖CoT提示的模型更安全，后者特别容易受到通过讲故事提示、虚构人物角色或奖励形状指令的上下文重组攻击。这些结果挑战了推理本质上可以提高稳健性的假设，并强调了对推理设计的更多偏差感知方法的需求。



## **25. The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions**

数据集蒸馏的演变：迈向可扩展和可推广的解决方案 cs.CV

Dr. Jiawei Du is the corresponding author

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2502.05673v3) [paper-pdf](http://arxiv.org/pdf/2502.05673v3)

**Authors**: Ping Liu, Jiawei Du

**Abstract**: Dataset distillation, which condenses large-scale datasets into compact synthetic representations, has emerged as a critical solution for training modern deep learning models efficiently. While prior surveys focus on developments before 2023, this work comprehensively reviews recent advances, emphasizing scalability to large-scale datasets such as ImageNet-1K and ImageNet-21K. We categorize progress into a few key methodologies: trajectory matching, gradient matching, distribution matching, scalable generative approaches, and decoupling optimization mechanisms. As a comprehensive examination of recent dataset distillation advances, this survey highlights breakthrough innovations: the SRe2L framework for efficient and effective condensation, soft label strategies that significantly enhance model accuracy, and lossless distillation techniques that maximize compression while maintaining performance. Beyond these methodological advancements, we address critical challenges, including robustness against adversarial and backdoor attacks, effective handling of non-IID data distributions. Additionally, we explore emerging applications in video and audio processing, multi-modal learning, medical imaging, and scientific computing, highlighting its domain versatility. By offering extensive performance comparisons and actionable research directions, this survey equips researchers and practitioners with practical insights to advance efficient and generalizable dataset distillation, paving the way for future innovations.

摘要: 数据集蒸馏将大规模数据集浓缩为紧凑的合成表示，已成为有效训练现代深度学习模型的关键解决方案。虽然之前的调查重点关注2023年之前的发展，但这项工作全面回顾了最近的进展，强调了ImageNet-1 K和ImageNet-21 K等大规模数据集的可扩展性。我们将进展分为几种关键方法：轨迹匹配、梯度匹配、分布匹配、可扩展生成方法和脱钩优化机制。作为对最近数据集蒸馏进展的全面检查，这项调查强调了突破性创新：用于高效浓缩的SRe 2L框架、显着提高模型准确性的软标签策略，以及在保持性能的同时最大限度地提高压缩率的无损蒸馏技术。除了这些方法论进步之外，我们还解决了关键挑战，包括对抗性和后门攻击的稳健性、有效处理非IID数据分布。此外，我们还探索视频和音频处理、多模式学习、医学成像和科学计算等领域的新兴应用，突出其领域的多功能性。通过提供广泛的性能比较和可操作的研究方向，这项调查为研究人员和从业者提供了实用见解，以推进高效且可推广的数据集提炼，为未来的创新铺平道路。



## **26. De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks**

De-AntiFake：重新思考针对语音克隆攻击的保护性干扰 cs.SD

Accepted by ICML 2025

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02606v1) [paper-pdf](http://arxiv.org/pdf/2507.02606v1)

**Authors**: Wei Fan, Kejiang Chen, Chang Liu, Weiming Zhang, Nenghai Yu

**Abstract**: The rapid advancement of speech generation models has heightened privacy and security concerns related to voice cloning (VC). Recent studies have investigated disrupting unauthorized voice cloning by introducing adversarial perturbations. However, determined attackers can mitigate these protective perturbations and successfully execute VC. In this study, we conduct the first systematic evaluation of these protective perturbations against VC under realistic threat models that include perturbation purification. Our findings reveal that while existing purification methods can neutralize a considerable portion of the protective perturbations, they still lead to distortions in the feature space of VC models, which degrades the performance of VC. From this perspective, we propose a novel two-stage purification method: (1) Purify the perturbed speech; (2) Refine it using phoneme guidance to align it with the clean speech distribution. Experimental results demonstrate that our method outperforms state-of-the-art purification methods in disrupting VC defenses. Our study reveals the limitations of adversarial perturbation-based VC defenses and underscores the urgent need for more robust solutions to mitigate the security and privacy risks posed by VC. The code and audio samples are available at https://de-antifake.github.io.

摘要: 语音生成模型的快速发展加剧了与语音克隆（VC）相关的隐私和安全问题。最近的研究调查了通过引入对抗性干扰来破坏未经授权的语音克隆。然而，坚定的攻击者可以减轻这些保护性干扰并成功执行VC。在这项研究中，我们在包括扰动净化在内的现实威胁模型下对这些针对VC的保护性扰动进行了首次系统评估。我们的研究结果表明，虽然现有的净化方法可以中和相当一部分的保护性扰动，但它们仍然会导致VC模型特征空间的失真，从而降低VC的性能。从这个角度出发，我们提出了一种新颖的两阶段净化方法：（1）净化受干扰的语音;（2）使用音素引导对其进行细化，使其与干净的语音分布保持一致。实验结果表明，我们的方法在破坏VC防御方面优于最先进的纯化方法。我们的研究揭示了基于对抗干扰的风险投资防御的局限性，并强调迫切需要更强大的解决方案来减轻风险投资带来的安全和隐私风险。代码和音频样本可在https://de-antifake.github.io上获取。



## **27. Robustness of Misinformation Classification Systems to Adversarial Examples Through BeamAttack**

错误信息分类系统对BeamAttack对抗示例的鲁棒性 cs.CL

12 pages main text, 27 pages total including references and  appendices. 13 figures, 10 tables. Accepted for publication in the LNCS  proceedings of CLEF 2025 (Best-of-Labs track)

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2506.23661v2) [paper-pdf](http://arxiv.org/pdf/2506.23661v2)

**Authors**: Arnisa Fazla, Lucas Krauter, David Guzman Piedrahita, Andrianos Michail

**Abstract**: We extend BeamAttack, an adversarial attack algorithm designed to evaluate the robustness of text classification systems through word-level modifications guided by beam search. Our extensions include support for word deletions and the option to skip substitutions, enabling the discovery of minimal modifications that alter model predictions. We also integrate LIME to better prioritize word replacements. Evaluated across multiple datasets and victim models (BiLSTM, BERT, and adversarially trained RoBERTa) within the BODEGA framework, our approach achieves over a 99\% attack success rate while preserving the semantic and lexical similarity of the original texts. Through both quantitative and qualitative analysis, we highlight BeamAttack's effectiveness and its limitations. Our implementation is available at https://github.com/LucK1Y/BeamAttack

摘要: 我们扩展了BeamAttack，这是一种对抗攻击算法，旨在通过束搜索指导的词级修改来评估文本分类系统的稳健性。我们的扩展包括对字词删除和跳过替换的选项，从而能够发现改变模型预测的最小修改。我们还集成了LIME，以更好地优先考虑单词替换。在BODEGA框架内对多个数据集和受害者模型（BiLSTM、BERT和对抗训练的RoBERTa）进行评估，我们的方法实现了超过99%的攻击成功率，同时保留了原始文本的语义和词汇相似性。通过定量和定性分析，我们强调了BeamAttack的有效性及其局限性。我们的实施可在https://github.com/LucK1Y/BeamAttack上获取



## **28. SecAlign: Defending Against Prompt Injection with Preference Optimization**

SecAlign：通过偏好优化抵御提示注入 cs.CR

ACM CCS 2025. Key words: prompt injection defense, LLM security,  LLM-integrated applications

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2410.05451v3) [paper-pdf](http://arxiv.org/pdf/2410.05451v3)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the Internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be injected into external data sources to override the system's intended instruction and instead execute a malicious instruction. To mitigate this vulnerability, we propose a new defense called SecAlign based on the technique of preference optimization. Our defense first constructs a preference dataset with prompt-injected inputs, secure outputs (ones that respond to the legitimate instruction), and insecure outputs (ones that respond to the injection). We then perform preference optimization on this dataset to teach the LLM to prefer the secure output over the insecure one. This provides the first known method that reduces the success rates of various prompt injections to <10%, even against attacks much more sophisticated than ones seen during training. This indicates our defense generalizes well against unknown and yet-to-come attacks. Also, SecAlign models are still practical with similar utility to the one before defensive training in our evaluations. Our code is at https://github.com/facebookresearch/SecAlign

摘要: 大型语言模型（LLM）在现代软件系统中变得越来越普遍，在用户和互联网之间进行接口，以协助执行需要高级语言理解的任务。为了完成这些任务，LLM通常使用外部数据源，例如用户文档、Web检索、API调用的结果等。这为攻击者通过提示注入操纵LLM开辟了新的途径。对抗性提示可以被注入到外部数据源中，以覆盖系统的预期指令，转而执行恶意指令。为了缓解此漏洞，我们基于偏好优化技术提出了一种名为SecAlign的新防御。我们的防御首先构建一个具有预算注入的输入、安全输出（响应合法指令的输出）和不安全输出（响应注入的输出）的偏好数据集。然后，我们对该数据集执行偏好优化，以教导LLM更喜欢安全的输出而不是不安全的输出。这提供了第一种已知的方法，可以将各种即时注射的成功率降低到<10%，即使是针对比训练期间看到的攻击复杂得多的攻击。这表明我们的防御对于未知和尚未到来的攻击具有很好的概括性。此外，SecAlign模型仍然实用，与我们评估中防御训练前的模型相似。我们的代码位于https://github.com/facebookresearch/SecAlign



## **29. Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability**

用于增强对抗可移植性的语义结构感知生成攻击 cs.CV

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2506.18248v2) [paper-pdf](http://arxiv.org/pdf/2506.18248v2)

**Authors**: Jongoh Jeong, Hunmin Yang, Jaeseok Jeong, Kuk-Jin Yoon

**Abstract**: Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR).

摘要: 生成性对抗攻击在白盒代理模型上训练扰动生成器，然后将精心设计的扰动应用于不可见的黑匣子受害者模型。与迭代攻击相比，这些方法提供了卓越的推理时效率、可扩展性和可移植性;然而，到目前为止，现有的研究尚未充分利用生成模型的表示能力来保存和利用语义信息。具体来说，生成器的中间激活编码了丰富的语义特征--对象边界和粗糙形状--这些特征仍然没有得到充分利用，从而限制了扰动与对象突出区域的对齐，而这些区域对于对抗性的可转移性至关重要。为了解决这个问题，我们引入了一个基于Mean Teacher的语义结构感知攻击框架，该框架充当时间平滑的特征参考。通过这个平滑的引用，我们通过特征提炼进一步指导学生的早期层激活和语义丰富的教师的早期层激活之间的语义一致性。通过基于经验发现将扰动合成锚定到生成器内语义突出的早期中间块，我们的方法引导对区域进行渐进对抗扰动，从而大大增强对抗转移性。我们对不同的模型、领域和任务进行了广泛的实验，以展示相对于最先进的生成性攻击的一致改进，并使用传统指标和我们新提出的意外纠正率（OCR）进行了全面评估。



## **30. Boosting Adversarial Transferability Against Defenses via Multi-Scale Transformation**

通过多规模转型提高针对防守的对抗转移能力 cs.CV

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01791v1) [paper-pdf](http://arxiv.org/pdf/2507.01791v1)

**Authors**: Zihong Guo, Chen Wan, Yayin Zheng, Hailing Kuang, Xiaohai Lu

**Abstract**: The transferability of adversarial examples poses a significant security challenge for deep neural networks, which can be attacked without knowing anything about them. In this paper, we propose a new Segmented Gaussian Pyramid (SGP) attack method to enhance the transferability, particularly against defense models. Unlike existing methods that generally focus on single-scale images, our approach employs Gaussian filtering and three types of downsampling to construct a series of multi-scale examples. Then, the gradients of the loss function with respect to each scale are computed, and their average is used to determine the adversarial perturbations. The proposed SGP can be considered an input transformation with high extensibility that is easily integrated into most existing adversarial attacks. Extensive experiments demonstrate that in contrast to the state-of-the-art methods, SGP significantly enhances attack success rates against black-box defense models, with average attack success rates increasing by 2.3% to 32.6%, based only on transferability.

摘要: 对抗性示例的可移植性对深度神经网络构成了重大的安全挑战，深度神经网络可能会在不了解它们的情况下受到攻击。在本文中，我们提出了一种新的分段高斯金字塔（SGP）攻击方法来增强可移植性，特别是针对防御模型。与通常关注单尺度图像的现有方法不同，我们的方法采用高斯过滤和三种类型的下采样来构建一系列多尺度示例。然后，计算损失函数相对于每个尺度的梯度，并使用其平均值来确定对抗性扰动。提出的SGP可以被认为是具有高扩展性的输入转换，可以轻松集成到大多数现有的对抗攻击中。大量实验表明，与最先进的方法相比，SGP显着提高了针对黑匣子防御模型的攻击成功率，仅基于可移植性，平均攻击成功率增加了2.3%至32.6%。



## **31. Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training**

没有偷看的调整：LLM后培训的可证明隐私和泛化边界 cs.LG

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01752v1) [paper-pdf](http://arxiv.org/pdf/2507.01752v1)

**Authors**: Ismail Labiad, Mathurin Videau, Matthieu Kowalski, Marc Schoenauer, Alessandro Leite, Julia Kempe, Olivier Teytaud

**Abstract**: Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, its reliance on large volumes of labeled data raises privacy and security concerns such as susceptibility to data poisoning attacks and the risk of overfitting. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. However, black box methods also pose significant challenges, including poor scalability to high-dimensional parameter spaces, as prevalent in large language models (LLMs), and high computational costs due to reliance on numerous model evaluations. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide strong theoretical bounds on generalization, differential privacy, susceptibility to data poisoning attacks, and robustness to extraction attacks. BBoxER operates on top of pre-trained LLMs, offering a lightweight and modular enhancement suitable for deployment in restricted or privacy-sensitive environments, in addition to non-vacuous generalization guarantees. In experiments with LLMs, we demonstrate empirically that Retrofitting methods are able to learn, showing how a few iterations of BBoxER improve performance and generalize well on a benchmark of reasoning datasets. This positions BBoxER as an attractive add-on on top of gradient-based optimization.

摘要: 基于对象的优化是深度学习的主力，通过反向传播提供高效且可扩展的训练。然而，它对大量标记数据的依赖引发了隐私和安全问题，例如容易受到数据中毒攻击和过度匹配的风险。相比之下，黑匣子优化方法将模型视为一个不透明的函数，仅依赖函数评估来指导优化，在数据访问受到限制、对抗风险较高或过度匹配令人担忧的场景中提供了一种有希望的替代方案。然而，黑匣子方法也带来了重大挑战，包括大型语言模型（LLM）中普遍存在的对多维参数空间的可扩展性较差，以及由于依赖大量模型评估而导致的高计算成本。本文介绍了BBoxER，这是一种用于LLM后训练的进化黑匣子方法，通过隐式压缩训练数据来引发信息瓶颈。利用信息流的可追溯性，我们在概括性、差异隐私、对数据中毒攻击的敏感性以及对提取攻击的鲁棒性方面提供了强大的理论界限。BBoxER在预先培训的LLM之上运行，除了非空洞的通用保证外，还提供适合在受限制或隐私敏感环境中部署的轻量级模块化增强。在LLM的实验中，我们经验地证明了Retrofit方法能够学习，展示了BBoxER的几次迭代如何提高性能并在推理数据集的基准上很好地概括。这使得BBoxER成为基于梯度的优化之上的一个有吸引力的附加组件。



## **32. Blockchain Address Poisoning**

区块链地址中毒 cs.CR

To appear in Proceedings of the 34th USENIX Security Symposium  (USENIX Security'25)

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2501.16681v3) [paper-pdf](http://arxiv.org/pdf/2501.16681v3)

**Authors**: Taro Tsuchiya, Jin-Dong Dong, Kyle Soska, Nicolas Christin

**Abstract**: In many blockchains, e.g., Ethereum, Binance Smart Chain (BSC), the primary representation used for wallet addresses is a hardly memorable 40-digit hexadecimal string. As a result, users often select addresses from their recent transaction history, which enables blockchain address poisoning. The adversary first generates lookalike addresses similar to one with which the victim has previously interacted, and then engages with the victim to ``poison'' their transaction history. The goal is to have the victim mistakenly send tokens to the lookalike address, as opposed to the intended recipient. Compared to contemporary studies, this paper provides four notable contributions. First, we develop a detection system and perform measurements over two years on both Ethereum and BSC. We identify 13~times more attack attempts than reported previously -- totaling 270M on-chain attacks targeting 17M victims. 6,633 incidents have caused at least 83.8M USD in losses, which makes blockchain address poisoning one of the largest cryptocurrency phishing schemes observed in the wild. Second, we analyze a few large attack entities using improved clustering techniques, and model attacker profitability and competition. Third, we reveal attack strategies -- targeted populations, success conditions (address similarity, timing), and cross-chain attacks. Fourth, we mathematically define and simulate the lookalike address generation process across various software- and hardware-based implementations, and identify a large-scale attacker group that appears to use GPUs. We also discuss defensive countermeasures.

摘要: 在许多区块链中，例如，以太坊，币安智能链（BSC），用于钱包地址的主要表示是一个几乎令人难忘的40位十六进制字符串。因此，用户经常从最近的交易历史记录中选择地址，这导致区块链地址中毒。对手首先生成与受害者之前互动过的地址相似的地址，然后与受害者互动以“毒害”他们的交易历史记录。目标是让受害者错误地将代币发送到外观相似的地址，而不是预期的收件人。与当代研究相比，本文提供了四个值得注意的贡献。首先，我们开发一个检测系统，并在两年内对以太坊和BSC进行测量。我们发现的攻击尝试比之前报告的多13倍--总计针对1700万受害者的2.7亿次链上攻击。6，633起事件已造成至少8，380万美元的损失，这使得区块链地址中毒成为野外观察到的最大的加密货币网络钓鱼计划之一。其次，我们使用改进的集群技术分析一些大型攻击实体，并对攻击者的盈利能力和竞争进行建模。第三，我们揭示了攻击策略--目标人群、成功条件（地址相似性、时机）和跨链攻击。第四，我们以数学方式定义和模拟各种基于软件和硬件的实现中相似的地址生成过程，并识别似乎使用图形处理器的大规模攻击者群体。我们还讨论防御对策。



## **33. Graph Representation-based Model Poisoning on Federated LLMs in CyberEdge Networks**

CyberEdge网络中联邦LLM上基于图表示的模型中毒 cs.CR

7 pages, 5 figures

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01694v1) [paper-pdf](http://arxiv.org/pdf/2507.01694v1)

**Authors**: Hanlin Cai, Haofan Dong, Houtianfu Wang, Kai Li, Ozgur B. Akan

**Abstract**: Federated large language models (FedLLMs) provide powerful generative capabilities in CyberEdge networks while protecting data privacy. However, FedLLMs remains highly vulnerable to model poisoning attacks. This article first reviews recent model poisoning techniques and existing defense mechanisms for FedLLMs, highlighting critical limitations, particularly under non-IID text distributions. In particular, current defenses primarily utilize distance-based outlier detection or norm constraints, operating under the assumption that adversarial updates significantly diverge from benign statistics. This assumption can fail when facing adaptive attackers targeting billionparameter LLMs. Next, this article investigates emerging Graph Representation-Based Model Poisoning (GRMP), a novel attack paradigm that leverages higher-order correlations among honest client gradients to synthesize malicious updates indistinguishable from legitimate model updates. GRMP can effectively evade advanced defenses, resulting in substantial accuracy loss and performance degradation. Moreover, this article outlines a research roadmap emphasizing the importance of graph-aware secure aggregation methods, FedLLMs-specific vulnerability metrics, and evaluation frameworks to strengthen the robustness of future federated language model deployments.

摘要: 联合大型语言模型（FedLLM）在CyberEdge网络中提供强大的生成能力，同时保护数据隐私。然而，FedLLM仍然极易受到模型中毒攻击。本文首先回顾了FedLLM最近的模型中毒技术和现有的防御机制，强调了关键的局限性，特别是在非IID文本分发下。特别是，当前的防御主要利用基于距离的离群值检测或规范约束，在对抗性更新与良性统计数据显着偏离的假设下运行。当面对针对十亿参数LLM的自适应攻击者时，这一假设可能会失败。接下来，本文研究了新兴的基于图表示的模型中毒（GRMP），这是一种新型攻击范式，它利用诚实客户端梯度之间的更高层相关性来合成与合法模型更新没有区别的恶意更新。GRMP可以有效规避高级防御，导致准确性大幅损失和性能下降。此外，本文还概述了一份研究路线图，强调图形感知的安全聚合方法、特定于FedLLM的漏洞指标和评估框架的重要性，以加强未来联邦语言模型部署的稳健性。



## **34. Learned-Database Systems Security**

学习数据库系统安全 cs.CR

Accepted at TMLR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2212.10318v4) [paper-pdf](http://arxiv.org/pdf/2212.10318v4)

**Authors**: Roei Schuster, Jin Peng Zhou, Thorsten Eisenhofer, Paul Grubbs, Nicolas Papernot

**Abstract**: A learned database system uses machine learning (ML) internally to improve performance. We can expect such systems to be vulnerable to some adversarial-ML attacks. Often, the learned component is shared between mutually-distrusting users or processes, much like microarchitectural resources such as caches, potentially giving rise to highly-realistic attacker models. However, compared to attacks on other ML-based systems, attackers face a level of indirection as they cannot interact directly with the learned model. Additionally, the difference between the attack surface of learned and non-learned versions of the same system is often subtle. These factors obfuscate the de-facto risks that the incorporation of ML carries. We analyze the root causes of potentially-increased attack surface in learned database systems and develop a framework for identifying vulnerabilities that stem from the use of ML. We apply our framework to a broad set of learned components currently being explored in the database community. To empirically validate the vulnerabilities surfaced by our framework, we choose 3 of them and implement and evaluate exploits against these. We show that the use of ML cause leakage of past queries in a database, enable a poisoning attack that causes exponential memory blowup in an index structure and crashes it in seconds, and enable index users to snoop on each others' key distributions by timing queries over their own keys. We find that adversarial ML is an universal threat against learned components in database systems, point to open research gaps in our understanding of learned-systems security, and conclude by discussing mitigations, while noting that data leakage is inherent in systems whose learned component is shared between multiple parties.

摘要: 学习数据库系统在内部使用机器学习（ML）来提高性能。我们可以预计此类系统很容易受到一些对抗ML攻击。通常，学习到的组件在相互不信任的用户或流程之间共享，就像缓存等微架构资源一样，可能会产生高度真实的攻击者模型。然而，与对其他基于ML的系统的攻击相比，攻击者面临一定程度的间接性，因为他们无法与学习模型直接交互。此外，同一系统的学习版本和非学习版本的攻击表面之间的差异通常很微妙。这些因素混淆了合并ML所带来的事实风险。我们分析了学习数据库系统中攻击面可能增加的根本原因，并开发了一个框架来识别源于ML的使用的漏洞。我们将我们的框架应用于数据库社区目前正在探索的一系列广泛的学习组件。为了从经验上验证我们的框架中出现的漏洞，我们选择了其中3个漏洞，并针对这些漏洞实施和评估利用。我们表明，ML的使用会导致数据库中过去的查询泄露，引发中毒攻击，导致索引结构中的指数级内存爆炸并在几秒钟内使其崩溃，并使索引用户能够通过对自己的密钥进行计时来窥探彼此的密钥分布。我们发现对抗性ML是针对数据库系统中学习组件的普遍威胁，指出了我们对学习系统安全性的理解中存在的研究差距，并通过讨论缓解措施来得出结论，同时指出数据泄露是其学习组件在多方之间共享的系统中固有的。



## **35. Slot: Provenance-Driven APT Detection through Graph Reinforcement Learning**

插槽：通过图强化学习进行源驱动APT检测 cs.CR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2410.17910v3) [paper-pdf](http://arxiv.org/pdf/2410.17910v3)

**Authors**: Wei Qiao, Yebo Feng, Teng Li, Zhuo Ma, Yulong Shen, JianFeng Ma, Yang Liu

**Abstract**: Advanced Persistent Threats (APTs) represent sophisticated cyberattacks characterized by their ability to remain undetected within the victim system for extended periods, aiming to exfiltrate sensitive data or disrupt operations. Existing detection approaches often struggle to effectively identify these complex threats, construct the attack chain for defense facilitation, or resist adversarial attacks. To overcome these challenges, we propose Slot, an advanced APT detection approach based on provenance graphs and graph reinforcement learning. Slot excels in uncovering multi-level hidden relationships, such as causal, contextual, and indirect connections, among system behaviors through provenance graph mining. By pioneering the integration of graph reinforcement learning, Slot dynamically adapts to new user activities and evolving attack strategies, enhancing its resilience against adversarial attacks. Additionally, Slot automatically constructs the attack chain according to detected attacks with clustering algorithms, providing precise identification of attack paths and facilitating the development of defense strategies. Evaluations with real-world datasets demonstrate Slot's outstanding accuracy, efficiency, adaptability, and robustness in APT detection, with most metrics surpassing state-of-the-art methods. Additionally, case studies conducted to assess Slot's effectiveness in supporting APT defense further establish it as a practical and reliable tool for cybersecurity protection.

摘要: 高级持续性威胁（APT）代表复杂的网络攻击，其特征是能够在受害者系统中长时间不被发现，旨在泄露敏感数据或扰乱运营。现有的检测方法常常难以有效识别这些复杂的威胁、构建防御促进的攻击链或抵抗对抗性攻击。为了克服这些挑战，我们提出了Slot，这是一种基于出处图和图强化学习的高级APT检测方法。Slot擅长通过出处图挖掘发现系统行为之间的多层隐藏关系，例如因果关系、上下文关系和间接联系。通过开创图强化学习的集成，Slot动态适应新的用户活动和不断发展的攻击策略，增强其对对抗性攻击的弹性。此外，Slot还根据检测到的攻击，通过集群算法自动构建攻击链，提供攻击路径的精确识别，促进防御策略的制定。对现实世界数据集的评估证明了Slot在APT检测方面出色的准确性、效率、适应性和稳健性，大多数指标都超越了最先进的方法。此外，为评估Slot支持APT防御的有效性而进行的案例研究进一步确立了其作为网络安全保护实用且可靠的工具的地位。



## **36. DARTS: A Dual-View Attack Framework for Targeted Manipulation in Federated Sequential Recommendation**

DARTS：联邦顺序推荐中针对性操纵的双视图攻击框架 cs.IR

10 pages. arXiv admin note: substantial text overlap with  arXiv:2409.07500; text overlap with arXiv:2212.05399 by other authors

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01383v1) [paper-pdf](http://arxiv.org/pdf/2507.01383v1)

**Authors**: Qitao Qin, Yucong Luo, Zhibo Chu

**Abstract**: Federated recommendation (FedRec) preserves user privacy by enabling decentralized training of personalized models, but this architecture is inherently vulnerable to adversarial attacks. Significant research has been conducted on targeted attacks in FedRec systems, motivated by commercial and social influence considerations. However, much of this work has largely overlooked the differential robustness of recommendation models. Moreover, our empirical findings indicate that existing targeted attack methods achieve only limited effectiveness in Federated Sequential Recommendation(FSR) tasks. Driven by these observations, we focus on investigating targeted attacks in FSR and propose a novel dualview attack framework, named DV-FSR. This attack method uniquely combines a sampling-based explicit strategy with a contrastive learning-based implicit gradient strategy to orchestrate a coordinated attack. Additionally, we introduce a specific defense mechanism tailored for targeted attacks in FSR, aiming to evaluate the mitigation effects of the attack method we proposed. Extensive experiments validate the effectiveness of our proposed approach on representative sequential models. Our codes are publicly available.

摘要: 联合推荐（FedRec）通过支持个性化模型的去中心化训练来保护用户隐私，但这种架构本质上很容易受到对抗攻击。出于商业和社会影响考虑，人们对FedRec系统中的定向攻击进行了大量研究。然而，这项工作的大部分内容在很大程度上忽视了推荐模型的差异稳健性。此外，我们的经验研究结果表明，现有的有针对性的攻击方法在联合顺序推荐（FSR）任务中仅实现有限的有效性。在这些观察的推动下，我们专注于调查FSR中的有针对性的攻击，并提出了一种新颖的双视图攻击框架，名为DV-FSR。这种攻击方法独特地将基于采样的显式策略与基于对比学习的隐式梯度策略结合起来，以协调一致的攻击。此外，我们还引入了一种针对FSR中的定向攻击量身定制的特定防御机制，旨在评估我们提出的攻击方法的缓解效果。大量实验验证了我们提出的方法对代表性序列模型的有效性。我们的代码是公开的。



## **37. 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation**

3D高斯飞溅驱动的多视图鲁棒物理对抗伪装生成 cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01367v1) [paper-pdf](http://arxiv.org/pdf/2507.01367v1)

**Authors**: Tianrui Lou, Xiaojun Jia, Siyuan Liang, Jiawei Liang, Ming Zhang, Yanjun Xiao, Xiaochun Cao

**Abstract**: Physical adversarial attack methods expose the vulnerabilities of deep neural networks and pose a significant threat to safety-critical scenarios such as autonomous driving. Camouflage-based physical attack is a more promising approach compared to the patch-based attack, offering stronger adversarial effectiveness in complex physical environments. However, most prior work relies on mesh priors of the target object and virtual environments constructed by simulators, which are time-consuming to obtain and inevitably differ from the real world. Moreover, due to the limitations of the backgrounds in training images, previous methods often fail to produce multi-view robust adversarial camouflage and tend to fall into sub-optimal solutions. Due to these reasons, prior work lacks adversarial effectiveness and robustness across diverse viewpoints and physical environments. We propose a physical attack framework based on 3D Gaussian Splatting (3DGS), named PGA, which provides rapid and precise reconstruction with few images, along with photo-realistic rendering capabilities. Our framework further enhances cross-view robustness and adversarial effectiveness by preventing mutual and self-occlusion among Gaussians and employing a min-max optimization approach that adjusts the imaging background of each viewpoint, helping the algorithm filter out non-robust adversarial features. Extensive experiments validate the effectiveness and superiority of PGA. Our code is available at:https://github.com/TRLou/PGA.

摘要: 物理对抗攻击方法暴露了深度神经网络的漏洞，并对自动驾驶等安全关键场景构成重大威胁。与基于补丁的攻击相比，基于伪装的物理攻击是一种更有前途的方法，可以在复杂的物理环境中提供更强的对抗效果。然而，大多数先前的工作依赖于目标对象的网格先验和模拟器构建的虚拟环境，获取这些先验很耗时，并且不可避免地与现实世界不同。此外，由于训练图像中背景的限制，以前的方法常常无法产生多视图鲁棒的对抗伪装，并且往往会陷入次优解决方案。由于这些原因，之前的工作缺乏针对不同观点和物理环境的对抗有效性和稳健性。我们提出了一种基于3D高斯飞溅（3DGS）的物理攻击框架，名为PGA，该框架只需少量图像即可提供快速精确的重建，并具有照片真实感的渲染能力。我们的框架通过防止高斯之间的相互遮挡和自遮挡，并采用调整每个观点的成像背景的最小-最大优化方法，帮助算法过滤掉非鲁棒的对抗特征，进一步增强了交叉视图的鲁棒性和对抗有效性。大量实验验证了PGA的有效性和优越性。我们的代码可访问：https://github.com/TRLou/PGA。



## **38. ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks**

ICLShield：探索和缓解上下文学习后门攻击 cs.LG

ICML 2025

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01321v1) [paper-pdf](http://arxiv.org/pdf/2507.01321v1)

**Authors**: Zhiyao Ren, Siyuan Liang, Aishan Liu, Dacheng Tao

**Abstract**: In-context learning (ICL) has demonstrated remarkable success in large language models (LLMs) due to its adaptability and parameter-free nature. However, it also introduces a critical vulnerability to backdoor attacks, where adversaries can manipulate LLM behaviors by simply poisoning a few ICL demonstrations. In this paper, we propose, for the first time, the dual-learning hypothesis, which posits that LLMs simultaneously learn both the task-relevant latent concepts and backdoor latent concepts within poisoned demonstrations, jointly influencing the probability of model outputs. Through theoretical analysis, we derive an upper bound for ICL backdoor effects, revealing that the vulnerability is dominated by the concept preference ratio between the task and the backdoor. Motivated by these findings, we propose ICLShield, a defense mechanism that dynamically adjusts the concept preference ratio. Our method encourages LLMs to select clean demonstrations during the ICL phase by leveraging confidence and similarity scores, effectively mitigating susceptibility to backdoor attacks. Extensive experiments across multiple LLMs and tasks demonstrate that our method achieves state-of-the-art defense effectiveness, significantly outperforming existing approaches (+26.02% on average). Furthermore, our method exhibits exceptional adaptability and defensive performance even for closed-source models (e.g., GPT-4).

摘要: 上下文学习（ICL）因其适应性和无参数性质而在大型语言模型（LLM）中取得了显着的成功。然而，它也引入了后门攻击的关键漏洞，对手可以通过简单地毒害一些ICL演示来操纵LLM行为。在本文中，我们首次提出了双重学习假设，该假设LLM同时学习与任务相关的潜在概念和中毒演示中的后门潜在概念，共同影响模型输出的可能性。通过理论分析，我们推导出ICL后门效应的上界，揭示了漏洞由任务与后门之间的概念偏好比决定。受这些发现的启发，我们提出了ICLShield，这是一种动态调整概念偏好比的防御机制。我们的方法鼓励LLM通过利用置信度和相似性分数在ICL阶段选择干净的演示，从而有效地降低对后门攻击的敏感性。跨多个LLM和任务的广泛实验表明，我们的方法实现了最先进的防御有效性，显着优于现有方法（平均+26.02%）。此外，即使对于闭源模型（例如，GPT-4）。



## **39. Defensive Adversarial CAPTCHA: A Semantics-Driven Framework for Natural Adversarial Example Generation**

防御性对抗验证码：用于自然对抗示例生成的语义驱动框架 cs.CV

13 pages, 6 figures

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2506.10685v3) [paper-pdf](http://arxiv.org/pdf/2506.10685v3)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Cong Wu, Tao Li, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: Traditional CAPTCHA (Completely Automated Public Turing Test to Tell Computers and Humans Apart) schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on the original image characteristics, resulting in distortions that hinder human interpretation and limit their applicability in scenarios where no initial input images are available. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (DAC), a novel framework that generates high-fidelity adversarial examples guided by attacker-specified semantics information. Leveraging a Large Language Model (LLM), DAC enhances CAPTCHA diversity and enriches the semantic information. To address various application scenarios, we examine the white-box targeted attack scenario and the black box untargeted attack scenario. For target attacks, we introduce two latent noise variables that are alternately guided in the diffusion step to achieve robust inversion. The synergy between gradient guidance and latent variable optimization achieved in this way ensures that the generated adversarial examples not only accurately align with the target conditions but also achieve optimal performance in terms of distributional consistency and attack effectiveness. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-DAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show that the defensive adversarial CAPTCHA generated by BP-DAC is able to defend against most of the unknown models, and the generated CAPTCHA is indistinguishable to both humans and DNNs.

摘要: 传统的CAPTCHA（完全自动化公共图灵测试来区分计算机和人类）计划越来越容易受到深度神经网络（DNN）支持的自动化攻击。现有的对抗攻击方法通常依赖于原始图像特征，从而导致失真，阻碍人类解释并限制其在没有初始输入图像可用的场景中的适用性。为了解决这些挑战，我们提出了无源对抗性验证码（ADC），这是一种新颖的框架，可以在攻击者指定的语义信息的指导下生成高保真对抗性示例。利用大型语言模型（LLM），DEC增强了CAPTCHA的多样性并丰富了语义信息。为了应对各种应用场景，我们研究了白盒定向攻击场景和黑匣子非定向攻击场景。对于目标攻击，我们引入了两个潜在噪音变量，它们在扩散步骤中交替引导，以实现鲁棒的反转。通过这种方式实现的梯度引导和潜在变量优化之间的协同作用，确保生成的对抗示例不仅与目标条件准确对齐，而且在分布一致性和攻击有效性方面实现最佳性能。在无目标攻击中，特别是对于黑匣子场景，我们引入了双路径无源对抗性CAPTCHA（BP-ADC），这是一种两步优化策略，采用多峰梯度和双路径优化来实现高效的误分类。实验表明，BP-ADC生成的防御性对抗CAPTCHA能够防御大多数未知模型，并且生成的CAPTCHA对于人类和DNN来说都无法区分。



## **40. CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs**

CAWLRY-V：一个用于视频MLLM对抗性攻击的大规模生成器框架 cs.CV

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00817v1) [paper-pdf](http://arxiv.org/pdf/2507.00817v1)

**Authors**: Jiaming Zhang, Rui Hu, Qing Guo, Wei Yang Bryan Lim

**Abstract**: Video Multimodal Large Language Models (V-MLLMs) have shown impressive capabilities in temporal reasoning and cross-modal understanding, yet their vulnerability to adversarial attacks remains underexplored due to unique challenges: complex cross-modal reasoning mechanisms, temporal dependencies, and computational constraints. We present CAVALRY-V (Cross-modal Language-Vision Adversarial Yielding for Videos), a novel framework that directly targets the critical interface between visual perception and language generation in V-MLLMs. Our approach introduces two key innovations: (1) a dual-objective semantic-visual loss function that simultaneously disrupts the model's text generation logits and visual representations to undermine cross-modal integration, and (2) a computationally efficient two-stage generator framework that combines large-scale pre-training for cross-model transferability with specialized fine-tuning for spatiotemporal coherence. Empirical evaluation on comprehensive video understanding benchmarks demonstrates that CAVALRY-V significantly outperforms existing attack methods, achieving 22.8% average improvement over the best baseline attacks on both commercial systems (GPT-4.1, Gemini 2.0) and open-source models (QwenVL-2.5, InternVL-2.5, Llava-Video, Aria, MiniCPM-o-2.6). Our framework achieves flexibility through implicit temporal coherence modeling rather than explicit regularization, enabling significant performance improvements even on image understanding (34.4% average gain). This capability demonstrates CAVALRY-V's potential as a foundational approach for adversarial research across multimodal systems.

摘要: 视频多模式大型语言模型（V-MLLM）在时态推理和跨模式理解方面表现出了令人印象深刻的能力，但由于独特的挑战，它们对对抗性攻击的脆弱性仍然没有得到充分的研究：复杂的跨模式推理机制、时态依赖性和计算限制。我们提出了CAWLRY-V（跨模式视觉对抗屈服视频），这是一个新颖的框架，直接针对V-MLLM中视觉感知和语言生成之间的关键界面。我们的方法引入了两个关键创新：（1）双目标语义视觉损失函数，它同时扰乱模型的文本生成日志和视觉表示以破坏跨模式集成，以及（2）计算高效的两阶段生成器框架，它将跨模型可移植性的大规模预训练与时空一致性的专门微调相结合。对全面视频理解基准的实证评估表明，CAWLRY-V的表现显着优于现有的攻击方法，比商业系统（GPT-4.1、Gemini 2.0）和开源模型（QwenVL-2.5、InternVL-2.5、Llava-Video、Aria、MiniCPM-o-2.6）的最佳基线攻击平均改进了22.8%。我们的框架通过隐式时间一致性建模而不是显式正规化来实现灵活性，即使在图像理解方面也能显着提高性能（平均提高34.4%）。这一能力展示了CAWLRY-V作为跨多模式系统对抗性研究的基础方法的潜力。



## **41. Cage-Based Deformation for Transferable and Undefendable Point Cloud Attack**

基于笼子的变形可转移和不可发现的点云攻击 cs.CV

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00690v1) [paper-pdf](http://arxiv.org/pdf/2507.00690v1)

**Authors**: Keke Tang, Ziyong Du, Weilong Peng, Xiaofei Wang, Peican Zhu, Ligang Liu, Zhihong Tian

**Abstract**: Adversarial attacks on point clouds often impose strict geometric constraints to preserve plausibility; however, such constraints inherently limit transferability and undefendability. While deformation offers an alternative, existing unstructured approaches may introduce unnatural distortions, making adversarial point clouds conspicuous and undermining their plausibility. In this paper, we propose CageAttack, a cage-based deformation framework that produces natural adversarial point clouds. It first constructs a cage around the target object, providing a structured basis for smooth, natural-looking deformation. Perturbations are then applied to the cage vertices, which seamlessly propagate to the point cloud, ensuring that the resulting deformations remain intrinsic to the object and preserve plausibility. Extensive experiments on seven 3D deep neural network classifiers across three datasets show that CageAttack achieves a superior balance among transferability, undefendability, and plausibility, outperforming state-of-the-art methods. Codes will be made public upon acceptance.

摘要: 对点云的对抗攻击通常会施加严格的几何约束以保持相似性;然而，此类约束本质上限制了可移植性和不可分割性。虽然变形提供了一种替代方案，但现有的非结构化方法可能会引入不自然的扭曲，使对抗性点云变得明显并破坏其合理性。在本文中，我们提出了CageAttack，这是一个基于笼子的变形框架，可以产生自然的对抗点云。它首先在目标物体周围构建一个笼子，为光滑、自然的变形提供结构化基础。然后将扰动应用于笼形点，其无缝传播到点云，确保产生的变形保持对象固有并保持相似性。对三个数据集的七个3D深度神经网络分类器进行的广泛实验表明，CageAttack在可移植性、不可预测性和可信性之间实现了卓越的平衡，优于最先进的方法。代码在接受后将公开。



## **42. How Resilient is QUIC to Security and Privacy Attacks?**

QUIC对安全和隐私攻击的弹性如何？ cs.CR

7 pages, 1 figure, 1 table

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2401.06657v3) [paper-pdf](http://arxiv.org/pdf/2401.06657v3)

**Authors**: Jayasree Sengupta, Debasmita Dey, Simone Ferlin-Reiter, Nirnay Ghosh, Vaibhav Bajpai

**Abstract**: QUIC has rapidly evolved into a cornerstone transport protocol for secure, low-latency communications, yet its deployment continues to expose critical security and privacy vulnerabilities, particularly during connection establishment phases and via traffic analysis. This paper systematically revisits a comprehensive set of attacks on QUIC and emerging privacy threats. Building upon these observations, we critically analyze recent IETF mitigation efforts, including TLS Encrypted Client Hello (ECH), Oblivious HTTP (OHTTP) and MASQUE. We analyze how these mechanisms enhance privacy while introducing new operational risks, particularly under adversarial load. Additionally, we discuss emerging challenges posed by post-quantum cryptographic (PQC) handshakes, including handshake expansion and metadata leakage risks. Our analysis highlights ongoing gaps between theoretical defenses and practical deployments, and proposes new research directions focused on adaptive privacy mechanisms. Building on these insights, we propose future directions to ensure long-term security of QUIC and aim to guide its evolution as a robust, privacy-preserving, and resilient transport foundation for the next-generation Internet.

摘要: QUIC已迅速发展成为安全、低延迟通信的基石传输协议，但其部署继续暴露关键的安全和隐私漏洞，特别是在连接建立阶段和流量分析期间。本文系统性地重新审视了针对QUIC和新出现的隐私威胁的一系列全面攻击。在这些观察的基础上，我们批判性地分析了最近的ETF缓解工作，包括SSL加密客户端Hello（ECH）、不经意的HTTP（Ohttp）和MASQUE。我们分析了这些机制如何增强隐私，同时引入新的运营风险，特别是在对抗负载下。此外，我们还讨论了后量子加密（PQC）握手带来的新挑战，包括握手扩展和元数据泄露风险。我们的分析强调了理论防御和实际部署之间持续存在的差距，并提出了专注于自适应隐私机制的新研究方向。在这些见解的基础上，我们提出了未来方向，以确保QUIC的长期安全，并旨在引导其发展成为下一代互联网的强大、保护隐私和弹性传输基础。



## **43. Lazarus Group Targets Crypto-Wallets and Financial Data while employing new Tradecrafts**

Lazarus Group瞄准加密钱包和金融数据，同时采用新的Tradecrafts cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2505.21725v2) [paper-pdf](http://arxiv.org/pdf/2505.21725v2)

**Authors**: Alessio Di Santo

**Abstract**: This report presents a comprehensive analysis of a malicious software sample, detailing its architecture, behavioral characteristics, and underlying intent. Through static and dynamic examination, the malware core functionalities, including persistence mechanisms, command-and-control communication, and data exfiltration routines, are identified and its supporting infrastructure is mapped. By correlating observed indicators of compromise with known techniques, tactics, and procedures, this analysis situates the sample within the broader context of contemporary threat campaigns and infers the capabilities and motivations of its likely threat actor.   Building on these findings, actionable threat intelligence is provided to support proactive defenses. Threat hunting teams receive precise detection hypotheses for uncovering latent adversarial presence, while monitoring systems can refine alert logic to detect anomalous activity in real time. Finally, the report discusses how this structured intelligence enhances predictive risk assessments, informs vulnerability prioritization, and strengthens organizational resilience against advanced persistent threats. By integrating detailed technical insights with strategic threat landscape mapping, this malware analysis report not only reconstructs past adversary actions but also establishes a robust foundation for anticipating and mitigating future attacks.

摘要: 本报告对恶意软件样本进行了全面分析，详细介绍了其架构、行为特征和潜在意图。通过静态和动态检查，识别恶意软件的核心功能，包括持久性机制、命令和控制通信以及数据溢出例程，并绘制其支持基础设施。通过将观察到的妥协指标与已知的技术、策略和程序关联起来，该分析将样本置于当代威胁活动的更广泛背景下，并推断其可能的威胁参与者的能力和动机。   在这些发现的基础上，提供可操作的威胁情报来支持主动防御。威胁搜寻团队接收精确的检测假设，以发现潜在的对抗存在，而监控系统可以完善警报逻辑以实时检测异常活动。最后，该报告讨论了这种结构化智能如何增强预测性风险评估、为漏洞优先排序提供信息以及加强组织应对高级持续威胁的弹性。通过将详细的技术见解与战略威胁格局映射集成，该恶意软件分析报告不仅重建了过去的对手行为，还为预测和减轻未来的攻击奠定了坚实的基础。



## **44. Plug. Play. Persist. Inside a Ready-to-Go Havoc C2 Infrastructure**

插头。玩吧坚持。准备就绪的Havoc C2基础设施内部 cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2507.00189v1) [paper-pdf](http://arxiv.org/pdf/2507.00189v1)

**Authors**: Alessio Di Santo

**Abstract**: This analysis focuses on a single Azure-hosted Virtual Machine at 52.230.23.114 that the adversary converted into an all-in-one delivery, staging and Command-and-Control node. The host advertises an out-of-date Apache 2.4.52 instance whose open directory exposes phishing lures, PowerShell loaders, Reflective Shell-Code, compiled Havoc Demon implants and a toolbox of lateral-movement binaries; the same server also answers on 8443/80 for encrypted beacon traffic. The web tier is riddled with publicly documented critical vulnerabilities, that would have allowed initial code-execution had the attackers not already owned the device.   Initial access is delivered through an HTML file that, once de-obfuscated, perfectly mimics Google Unusual sign-in attempt notification and funnels victims toward credential collection. A PowerShell command follows: it disables AMSI in-memory, downloads a Base64-encoded stub, allocates RWX pages and starts the shell-code without ever touching disk. That stub reconstructs a DLL in memory using the Reflective-Loader technique and hands control to Havoc Demon implant. Every Demon variant-32- and 64-bit alike-talks to the same backend, resolves Windows APIs with hashed look-ups, and hides its activity behind indirect syscalls.   Runtime telemetry shows interests in registry under Image File Execution Options, deliberate queries to Software Restriction Policy keys, and heavy use of Crypto DLLs to protect payloads and C2 traffic. The attacker toolkit further contains Chisel, PsExec, Doppelganger and Whisker, some of them re-compiled under user directories that leak the developer personas tonzking123 and thobt. Collectively the findings paint a picture of a technically adept actor who values rapid re-tooling over deep operational security, leaning on Havoc modularity and on legitimate cloud services to blend malicious flows into ordinary enterprise traffic.

摘要: 此分析重点关注52.230.23.114上的单个Azure托管虚拟机，对手将其转换为一体化交付、中转和命令与控制节点。主机通知一个过时的Apache 2.4.52实例，其打开目录暴露了网络钓鱼诱饵、PowerShell加载器、反射性Shell-Code、已编译的Havoc Demon植入物和侧向移动二进制文件工具箱;同一服务器还在8443/80上回答加密信标流量。Web层充满了公开记录的关键漏洞，如果攻击者尚未拥有该设备，这些漏洞将允许初始代码执行。   初始访问通过一个HTML文件进行，该文件一旦去模糊，就会完美模仿Google Unusual登录尝试通知，并将受害者引导到凭证收集中。下面是一个Shell命令：它禁用内存中的AMSI，下载Base 64编码的树桩，分配RWX页面并在不接触磁盘的情况下启动shell代码。该树桩使用反射加载器技术在内存中重建了一个动态链接库，并将控制权交给Havoc Demon植入物。每个Demon变体（32位和64位相似）都与相同的后台对话，通过哈希查找来解析Windows API，并将其活动隐藏在间接系统缩放后面。   收件箱遥感显示对图像文件执行选项下的注册表的兴趣、对软件限制策略密钥的故意查询以及大量使用加密DLC来保护有效负载和C2流量。攻击者工具包还包含Chisel、PsExec、Doppelganger和Whisker，其中一些是在泄露开发人员角色tonzking 123和thobt的用户目录下重新编译的。总的来说，这些调查结果描绘了一幅技术精湛的参与者的图景，他重视快速重组而不是深度运营安全，依靠Havoc模块化和合法的云服务将恶意流量混合到普通企业流量中。



## **45. SQUASH: A SWAP-Based Quantum Attack to Sabotage Hybrid Quantum Neural Networks**

SQUASH：一种基于交换的量子攻击，旨在破坏混合量子神经网络 quant-ph

Keywords: Quantum Machine Learning, Hybrid Quantum Neural Networks,  SWAP Test, Fidelity, Circuit-level Attack

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24081v1) [paper-pdf](http://arxiv.org/pdf/2506.24081v1)

**Authors**: Rahul Kumar, Wenqi Wei, Ying Mao, Junaid Farooq, Ying Wang, Juntao Chen

**Abstract**: We propose a circuit-level attack, SQUASH, a SWAP-Based Quantum Attack to sabotage Hybrid Quantum Neural Networks (HQNNs) for classification tasks. SQUASH is executed by inserting SWAP gate(s) into the variational quantum circuit of the victim HQNN. Unlike conventional noise-based or adversarial input attacks, SQUASH directly manipulates the circuit structure, leading to qubit misalignment and disrupting quantum state evolution. This attack is highly stealthy, as it does not require access to training data or introduce detectable perturbations in input states. Our results demonstrate that SQUASH significantly degrades classification performance, with untargeted SWAP attacks reducing accuracy by up to 74.08\% and targeted SWAP attacks reducing target class accuracy by up to 79.78\%. These findings reveal a critical vulnerability in HQNN implementations, underscoring the need for more resilient architectures against circuit-level adversarial interventions.

摘要: 我们提出了一种电路级攻击SQUASH，一种基于SWAP的量子攻击，以破坏用于分类任务的混合量子神经网络（HQNN）。SQUASH通过将SWAP门插入到受害者HQNN的变分量子电路中来执行。与传统的基于噪声或对抗性输入攻击不同，SQUASH直接操纵电路结构，导致量子比特错位并破坏量子态演化。这种攻击是高度隐蔽的，因为它不需要访问训练数据或在输入状态中引入可检测的扰动。我们的结果表明，SQUASH会显着降低分类性能，非目标SWAP攻击将准确性降低高达74.08%，而目标SWAP攻击将目标类准确性降低高达79.78%。这些发现揭示了HQNN实现中的一个关键漏洞，强调了对电路级对抗干预的更具弹性的架构的需要。



## **46. STACK: Adversarial Attacks on LLM Safeguard Pipelines**

STACK：对LLM Safeguard Pipelines的对抗性攻击 cs.CL

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24068v1) [paper-pdf](http://arxiv.org/pdf/2506.24068v1)

**Authors**: Ian R. McKenzie, Oskar J. Hollinsworth, Tom Tseng, Xander Davies, Stephen Casper, Aaron D. Tucker, Robert Kirk, Adam Gleave

**Abstract**: Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic guards their latest Claude 4 Opus model using one such defense pipeline, and other frontier developers including Google DeepMind and OpenAI pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks.

摘要: 前沿人工智能开发人员依靠多层保障措施来防止人工智能系统的灾难性滥用。Anthropic使用这样的防御管道来保护他们最新的Claude 4 Opus模型，包括Google DeepMind和OpenAI在内的其他前沿开发商承诺很快部署类似的防御。然而，此类管道的安全性尚不清楚，之前评估或攻击这些管道的工作有限。我们通过开发和组建开源防御管道来解决这一差距。首先，我们发现一种新型的几次激发输入和输出分类器在三次攻击和两个数据集中优于最先进的开权保护模型ShieldGemma，将灾难性滥用数据集ClearHarm的攻击成功率（ASO）降低至0%。其次，我们引入了一个STaged AttaCK（STACK）过程，该过程在ClearHarm上实现了71%的ASB，针对少量镜头提示的分类器管道进行黑匣子攻击。最后，我们还在传输环境中评估了STACK，实现了33%的ASB，提供了初步证据，证明在不访问目标管道的情况下设计攻击是可行的。最后，我们建议开发人员可以用来阻止分阶段攻击的具体缓解措施。



## **47. Consensus-based optimization for closed-box adversarial attacks and a connection to evolution strategies**

针对闭箱对抗攻击的基于假设的优化以及与进化策略的联系 math.OC

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24048v1) [paper-pdf](http://arxiv.org/pdf/2506.24048v1)

**Authors**: Tim Roith, Leon Bungert, Philipp Wacker

**Abstract**: Consensus-based optimization (CBO) has established itself as an efficient gradient-free optimization scheme, with attractive mathematical properties, such as mean-field convergence results for non-convex loss functions. In this work, we study CBO in the context of closed-box adversarial attacks, which are imperceptible input perturbations that aim to fool a classifier, without accessing its gradient. Our contribution is to establish a connection between the so-called consensus hopping as introduced by Riedl et al. and natural evolution strategies (NES) commonly applied in the context of adversarial attacks and to rigorously relate both methods to gradient-based optimization schemes. Beyond that, we provide a comprehensive experimental study that shows that despite the conceptual similarities, CBO can outperform NES and other evolutionary strategies in certain scenarios.

摘要: 基于边界的优化（CBO）已经成为一种高效的无梯度优化方案，具有吸引人的数学性质，例如非凸损失函数的平均场收敛结果。在这项工作中，我们在闭箱对抗攻击的背景下研究CBO，这是一种难以察觉的输入扰动，旨在欺骗分类器，而无需访问其梯度。我们的贡献是在Riedl等人提出的所谓共识跳跃与通常在对抗性攻击背景下应用的自然进化策略（NES）之间建立联系，并将这两种方法与基于梯度的优化方案严格联系起来。除此之外，我们还提供了一项全面的实验研究，表明尽管概念相似，但CBO在某些情况下可以优于NES和其他进化策略。



## **48. Quickest Detection of Adversarial Attacks Against Correlated Equilibria**

最快检测针对相关均衡的对抗攻击 cs.GT

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24040v1) [paper-pdf](http://arxiv.org/pdf/2506.24040v1)

**Authors**: Kiarash Kazari, Aris Kanellopoulos, György Dán

**Abstract**: We consider correlated equilibria in strategic games in an adversarial environment, where an adversary can compromise the public signal used by the players for choosing their strategies, while players aim at detecting a potential attack as soon as possible to avoid loss of utility. We model the interaction between the adversary and the players as a zero-sum game and we derive the maxmin strategies for both the defender and the attacker using the framework of quickest change detection. We define a class of adversarial strategies that achieve the optimal trade-off between attack impact and attack detectability and show that a generalized CUSUM scheme is asymptotically optimal for the detection of the attacks. Our numerical results on the Sioux-Falls benchmark traffic routing game show that the proposed detection scheme can effectively limit the utility loss by a potential adversary.

摘要: 我们考虑了在对抗环境中的策略博弈中的相关均衡，在这种环境中，对手可以损害玩家选择策略所使用的公共信号，而玩家的目标是尽快检测到潜在的攻击，以避免效用损失。我们建模的对手和球员之间的相互作用作为一个零和博弈，我们推导出最大最小的防御者和攻击者使用的框架，最快的变化检测的策略。我们定义了一类对抗策略，实现了攻击影响和攻击可检测性之间的最佳权衡，并证明了广义的CRAMUM方案对于检测攻击是渐近最优的。在Sioux-Falls基准流量路由博弈上的数值结果表明，该检测方案可以有效地限制潜在对手的效用损失。



## **49. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

谜语我！检索增强一代的隐形会员推断 cs.CR

This is the full version (27 pages) of the paper 'Riddle Me This!  Stealthy Membership Inference for Retrieval-Augmented Generation' published  at CCS 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2502.00306v2) [paper-pdf](http://arxiv.org/pdf/2502.00306v2)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.

摘要: 检索增强生成（RAG）使大型语言模型（LLM）能够通过利用外部知识数据库来生成接地响应，而无需更改模型参数。尽管缺乏权重调整可以防止模型参数泄露，但它引入了推理对手利用模型上下文中检索到的文档的风险。现有的隶属关系推断和数据提取方法通常依赖于越狱或精心制作的非自然查询，这些查询可以通过RAG系统中常见的查询重写技术轻松检测或阻止。在这项工作中，我们介绍了审讯攻击（IA），这是一种针对RAG收件箱中文档的成员资格推断技术。通过制作仅在目标文档存在的情况下才能回答的自然文本查询，我们的方法仅用30个查询就能证明成功推理，同时保持隐蔽性;简单的检测器识别来自现有方法的对抗性提示的频率高达约76倍，比我们的攻击产生的提示。我们观察到，在各种RAG配置中，TPR@1%FPR比之前的推理攻击提高了2倍，同时每个文档推理的成本不到0.02美元。



## **50. Benchmarking Spiking Neural Network Learning Methods with Varying Locality**

对具有不同局部性的尖峰神经网络学习方法进行基准测试 cs.NE

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2402.01782v2) [paper-pdf](http://arxiv.org/pdf/2402.01782v2)

**Authors**: Jiaqi Lin, Sen Lu, Malyaban Bal, Abhronil Sengupta

**Abstract**: Spiking Neural Networks (SNNs), providing more realistic neuronal dynamics, have been shown to achieve performance comparable to Artificial Neural Networks (ANNs) in several machine learning tasks. Information is processed as spikes within SNNs in an event-based mechanism that significantly reduces energy consumption. However, training SNNs is challenging due to the non-differentiable nature of the spiking mechanism. Traditional approaches, such as Backpropagation Through Time (BPTT), have shown effectiveness but come with additional computational and memory costs and are biologically implausible. In contrast, recent works propose alternative learning methods with varying degrees of locality, demonstrating success in classification tasks. In this work, we show that these methods share similarities during the training process, while they present a trade-off between biological plausibility and performance. Further, given the implicitly recurrent nature of SNNs, this research investigates the influence of the addition of explicit recurrence to SNNs. We experimentally prove that the addition of explicit recurrent weights enhances the robustness of SNNs. We also investigate the performance of local learning methods under gradient and non-gradient-based adversarial attacks.

摘要: 尖峰神经网络（SNN）提供了更真实的神经元动力学，已被证明在几项机器学习任务中可以实现与人工神经网络（ANN）相当的性能。信息在基于事件的机制中作为SNN内的尖峰进行处理，从而显着降低了能源消耗。然而，由于尖峰机制的不可微性质，训练SNN具有挑战性。传统方法，例如时间反向传播（BPTT），已经显示出有效性，但会带来额外的计算和存储成本，并且在生物学上是不可信的。相比之下，最近的作品提出了具有不同局部性的替代学习方法，证明了分类任务的成功。在这项工作中，我们表明这些方法在训练过程中有相似之处，同时它们在生物相似性和性能之间进行了权衡。此外，鉴于SNN的隐式回归性质，本研究调查了SNN添加显式回归的影响。我们通过实验证明，添加显式循环权重增强了SNN的鲁棒性。我们还研究了本地学习方法在梯度和非基于梯度的对抗攻击下的性能。




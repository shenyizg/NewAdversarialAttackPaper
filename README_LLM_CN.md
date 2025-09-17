# Latest Large Language Model Attack Papers
**update at 2025-09-17 09:51:53**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. TrojanRobot: Physical-world Backdoor Attacks Against VLM-based Robotic Manipulation**

TrojanRobot：针对基于VLM的机器人操纵的物理世界后门攻击 cs.RO

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.11683v5) [paper-pdf](http://arxiv.org/pdf/2411.11683v5)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Aishan Liu, Yunpeng Jiang, Leo Yu Zhang, Xiaohua Jia

**Abstract**: Robotic manipulation in the physical world is increasingly empowered by \textit{large language models} (LLMs) and \textit{vision-language models} (VLMs), leveraging their understanding and perception capabilities. Recently, various attacks against such robotic policies have been proposed, with backdoor attacks drawing considerable attention for their high stealth and strong persistence capabilities. However, existing backdoor efforts are limited to simulators and suffer from physical-world realization. To address this, we propose \textit{TrojanRobot}, a highly stealthy and broadly effective robotic backdoor attack in the physical world. Specifically, we introduce a module-poisoning approach by embedding a backdoor module into the modular robotic policy, enabling backdoor control over the policy's visual perception module thereby backdooring the entire robotic policy. Our vanilla implementation leverages a backdoor-finetuned VLM to serve as the backdoor module. To enhance its generalization in physical environments, we propose a prime implementation, leveraging the LVLM-as-a-backdoor paradigm and developing three types of prime attacks, \ie, \textit{permutation}, \textit{stagnation}, and \textit{intentional} attacks, thus achieving finer-grained backdoors. Extensive experiments on the UR3e manipulator with 18 task instructions using robotic policies based on four VLMs demonstrate the broad effectiveness and physical-world stealth of TrojanRobot. Our attack's video demonstrations are available via a github link https://trojanrobot.github.io.

摘要: \textit{大型语言模型}（LLM）和\textit{视觉语言模型}（VLMS）利用它们的理解和感知能力，越来越多地增强物理世界中的机器人操纵能力。最近，针对此类机器人策略的各种攻击被提出，其中后门攻击因其高隐身性和强持久性能力而引起了相当大的关注。然而，现有的后门工作仅限于模拟器，并且受到物理世界实现的影响。为了解决这个问题，我们提出了\textit{TrojanRobot}，这是物理世界中一种高度隐蔽且广泛有效的机器人后门攻击。具体来说，我们通过将后门模块嵌入模块到模块化机器人策略中来引入模块中毒方法，从而对策略的视觉感知模块进行后门控制，从而后门化整个机器人策略。我们的普通实现利用一个经过后门微调的VLM作为后门模块。为了增强其在物理环境中的通用性，我们提出了一种主要实现，利用LVLM作为后门范式并开发三种类型的主要攻击，即\textit{perspective}、\textit{staduction}和\textit{intentional}攻击，从而实现更细粒度的后门。UR 3e机械手与18个任务指令使用机器人策略的基础上，四个VLMs的广泛实验证明了广泛的有效性和物理世界的隐身TrojanRobot。我们的攻击视频演示可以通过github链接https://trojanrobot.github.io获得。



## **2. Context-Aware Membership Inference Attacks against Pre-trained Large Language Models**

针对预训练大型语言模型的上下文感知成员推断攻击 cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2409.13745v2) [paper-pdf](http://arxiv.org/pdf/2409.13745v2)

**Authors**: Hongyan Chang, Ali Shahin Shamsabadi, Kleomenis Katevas, Hamed Haddadi, Reza Shokri

**Abstract**: Membership Inference Attacks (MIAs) on pre-trained Large Language Models (LLMs) aim at determining if a data point was part of the model's training set. Prior MIAs that are built for classification models fail at LLMs, due to ignoring the generative nature of LLMs across token sequences. In this paper, we present a novel attack on pre-trained LLMs that adapts MIA statistical tests to the perplexity dynamics of subsequences within a data point. Our method significantly outperforms prior approaches, revealing context-dependent memorization patterns in pre-trained LLMs.

摘要: 对预训练的大型语言模型（LLM）的成员推断攻击（MIA）旨在确定数据点是否是模型训练集的一部分。先前为分类模型构建的MIA在LLM上失败，因为忽视了令牌序列之间的LLM的生成性质。在本文中，我们提出了一种对预训练的LLM的新型攻击，该攻击将MIA统计测试适应数据点内子序列的困惑动态。我们的方法显着优于之前的方法，揭示了预训练的LLM中依赖上下文的记忆模式。



## **3. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

大型多模式模型的鲁棒适应用于检索增强仇恨模因检测 cs.CL

EMNLP 2025 Main (Oral)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2502.13061v4) [paper-pdf](http://arxiv.org/pdf/2502.13061v4)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL

摘要: 仇恨模因已成为互联网上的一个重要问题，需要强大的自动化检测系统。虽然大型多模式模型（LSYS）在仇恨模因检测方面表现出了希望，但它们面临着显着的挑战，例如次优的性能和有限的域外概括能力。最近的研究进一步揭示了在这种环境下将监督微调（SFT）和上下文学习应用于LSYS时的局限性。为了解决这些问题，我们提出了一个用于仇恨模因检测的鲁棒适应框架，该框架可以增强领域内准确性和跨领域概括性，同时保留Letts的一般视觉语言能力。分析表明，与SFT模型相比，我们的方法在对抗攻击下实现了更好的鲁棒性。对六个模因分类数据集的实验表明，我们的方法实现了最先进的性能，优于更大的代理系统。此外，与标准SFT相比，我们的方法为解释仇恨内容生成了更高质量的理由，增强了模型的可解释性。代码可访问https://github.com/JingbiaoMei/RGCL



## **4. Adversarial Prompt Distillation for Vision-Language Models**

视觉语言模型的对抗性提示蒸馏 cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.15244v3) [paper-pdf](http://arxiv.org/pdf/2411.15244v3)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.

摘要: 对比图像预训练（CLIP）等大型预训练视觉语言模型（VLM）已被证明容易受到对抗攻击，这引发了人们对其在自动驾驶和医疗诊断等安全关键应用中部署的担忧。对抗性提示调整（APT）是对预训练的VLM进行鲁棒化的一种有希望的方法，它在提示调整的过程中应用对抗性训练。然而，现有的APT方法大多是单模式方法，仅为视觉或文本模式设计提示，从而限制了其稳健性或清晰准确性的有效性。在这项工作中，我们提出了对抗性提示蒸馏（APT），这是一个双峰知识蒸馏框架，通过将APT与多模式知识转移集成来增强APT。APT优化视觉和文本模式的提示，同时从干净的预培训教师CLIP模型中提取知识。对多个基准数据集的广泛实验证明了我们的APT方法在对抗稳健性和精确性方面优于当前最先进的APT方法。APT的有效性也验证了使用非稳健教师来提高微调后的VLM的通用性和稳健性的可能性。



## **5. DiffHash: Text-Guided Targeted Attack via Diffusion Models against Deep Hashing Image Retrieval**

迪夫哈希：通过针对深度哈希图像检索的扩散模型进行文本引导定向攻击 cs.IR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12824v1) [paper-pdf](http://arxiv.org/pdf/2509.12824v1)

**Authors**: Zechao Liu, Zheng Zhou, Xiangkun Chen, Tao Liang, Dapeng Lang

**Abstract**: Deep hashing models have been widely adopted to tackle the challenges of large-scale image retrieval. However, these approaches face serious security risks due to their vulnerability to adversarial examples. Despite the increasing exploration of targeted attacks on deep hashing models, existing approaches still suffer from a lack of multimodal guidance, reliance on labeling information and dependence on pixel-level operations for attacks. To address these limitations, we proposed DiffHash, a novel diffusion-based targeted attack for deep hashing. Unlike traditional pixel-based attacks that directly modify specific pixels and lack multimodal guidance, our approach focuses on optimizing the latent representations of images, guided by text information generated by a Large Language Model (LLM) for the target image. Furthermore, we designed a multi-space hash alignment network to align the high-dimension image space and text space to the low-dimension binary hash space. During reconstruction, we also incorporated text-guided attention mechanisms to refine adversarial examples, ensuring them aligned with the target semantics while maintaining visual plausibility. Extensive experiments have demonstrated that our method outperforms state-of-the-art (SOTA) targeted attack methods, achieving better black-box transferability and offering more excellent stability across datasets.

摘要: 深度哈希模型已被广泛采用来应对大规模图像检索的挑战。然而，由于这些方法容易受到对抗示例的影响，因此面临严重的安全风险。尽管人们越来越多地探索深度哈希模型的有针对性的攻击，但现有方法仍然缺乏多模式指导、依赖标签信息以及依赖像素级操作进行攻击。为了解决这些限制，我们提出了迪夫哈希，这是一种新型的基于扩散的深度哈希定向攻击。与直接修改特定像素且缺乏多模式指导的传统基于像素的攻击不同，我们的方法重点是优化图像的潜在表示，并由目标图像的大型语言模型（LLM）生成的文本信息指导。此外，我们设计了一个多空间哈希对齐网络，将多维图像空间和文本空间与低维二进制哈希空间对齐。在重建过程中，我们还结合了文本引导的注意力机制来完善对抗性示例，确保它们与目标语义保持一致，同时保持视觉可信性。大量实验表明，我们的方法优于最先进的（SOTA）定向攻击方法，实现了更好的黑匣子可转移性，并在数据集之间提供更出色的稳定性。



## **6. Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection**

视觉上下文攻击：利用图像驱动上下文注入越狱MLLM cs.CV

Accepted to EMNLP 2025 (Main). 17 pages, 7 figures

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2507.02844v2) [paper-pdf](http://arxiv.org/pdf/2507.02844v2)

**Authors**: Ziqi Miao, Yi Ding, Lijun Li, Jing Shao

**Abstract**: With the emergence of strong vision language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: vision-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct vision-focused strategies, dynamically generating auxiliary images when necessary to construct a vision-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which achieves a toxicity score of 2.48 and an ASR of 22.2%. Code: https://github.com/Dtc7w3PQ/Visco-Attack.

摘要: 随着强大视觉语言能力的出现，多模式大型语言模型（MLLM）在现实世界应用中展示了巨大的潜力。然而，视觉模式所表现出的安全漏洞对在开放世界环境中部署此类模型构成了重大挑战。最近的研究通过将有害的文本语义直接编码到视觉输入中，成功地诱导了目标MLLM的有害反应。然而，在这些方法中，视觉形态主要充当不安全行为的触发器，通常表现出语义模糊性并且在现实场景中缺乏基础。在这项工作中，我们定义了一种新颖的环境：以视觉为中心的越狱，其中视觉信息是构建完整而现实的越狱背景的必要组成部分。在此设置的基础上，我们提出了VisCo（视觉上下文）攻击。VisCo使用四种不同的以视觉为中心的策略构建上下文对话，在必要时动态生成辅助图像，以构建以视觉为中心的越狱场景。为了最大限度地提高攻击效果，它结合了自动毒性混淆和语义细化，以产生最终的攻击提示，从而可靠地触发目标黑匣子MLLM的有害响应。具体而言，VisCo在MM-SafetyBench上对GPT-4 o的毒性评分为4.78，攻击成功率（ASB）为85%，显着优于基线，基线达到了2.48的毒性评分和22.2%的ASB。代码：https://github.com/Dtc7w3PQ/Visco-Attack。



## **7. SoK: How Sensor Attacks Disrupt Autonomous Vehicles: An End-to-end Analysis, Challenges, and Missed Threats**

SoK：传感器攻击如何扰乱自动驾驶车辆：端到端分析、挑战和错过的威胁 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11120v2) [paper-pdf](http://arxiv.org/pdf/2509.11120v2)

**Authors**: Qingzhao Zhang, Shaocheng Luo, Z. Morley Mao, Miroslav Pajic, Michael K. Reiter

**Abstract**: Autonomous vehicles, including self-driving cars, robotic ground vehicles, and drones, rely on complex sensor pipelines to ensure safe and reliable operation. However, these safety-critical systems remain vulnerable to adversarial sensor attacks that can compromise their performance and mission success. While extensive research has demonstrated various sensor attack techniques, critical gaps remain in understanding their feasibility in real-world, end-to-end systems. This gap largely stems from the lack of a systematic perspective on how sensor errors propagate through interconnected modules in autonomous systems when autonomous vehicles interact with the physical world.   To bridge this gap, we present a comprehensive survey of autonomous vehicle sensor attacks across platforms, sensor modalities, and attack methods. Central to our analysis is the System Error Propagation Graph (SEPG), a structured demonstration tool that illustrates how sensor attacks propagate through system pipelines, exposing the conditions and dependencies that determine attack feasibility. With the aid of SEPG, our study distills seven key findings that highlight the feasibility challenges of sensor attacks and uncovers eleven previously overlooked attack vectors exploiting inter-module interactions, several of which we validate through proof-of-concept experiments. Additionally, we demonstrate how large language models (LLMs) can automate aspects of SEPG construction and cross-validate expert analysis, showcasing the promise of AI-assisted security evaluation.

摘要: 自动驾驶汽车、机器人地面车辆和无人机等自动驾驶车辆依赖复杂的传感器管道来确保安全可靠的运行。然而，这些安全关键系统仍然容易受到对抗性传感器攻击，这可能会损害其性能和任务成功。虽然广泛的研究已经证明了各种传感器攻击技术，但在了解其在现实世界的端到端系统中的可行性方面仍然存在重大差距。这一差距很大程度上源于缺乏对自动驾驶汽车与物理世界互动时传感器误差如何通过自动驾驶系统中的互连模块传播的系统视角。   为了弥合这一差距，我们对跨平台、传感器模式和攻击方法的自动驾驶汽车传感器攻击进行了全面调查。我们分析的核心是系统错误传播图（SEPG），这是一种结构化演示工具，它说明了传感器攻击如何通过系统管道传播，揭示了决定攻击可行性的条件和依赖性。在SEPG的帮助下，我们的研究提炼了七个关键发现，这些发现凸显了传感器攻击的可行性挑战，并揭示了11个以前被忽视的利用模块间交互的攻击载体，其中一些我们通过概念验证实验进行了验证。此外，我们还展示了大型语言模型（LLM）如何自动化SEPG构建的各个方面和交叉验证专家分析，展示了人工智能辅助安全评估的前景。



## **8. Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content**

迈向包容性有毒内容审核：解决毒性分类器中对抗攻击的脆弱性，以应对LLM生成的内容 cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12672v1) [paper-pdf](http://arxiv.org/pdf/2509.12672v1)

**Authors**: Shaz Furniturewala, Arkaitz Zubiaga

**Abstract**: The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.

摘要: 由于大型语言模型（LLM）的广泛使用，在线机器生成内容的数量急剧增长，这给内容审核系统带来了新的挑战。传统的内容审核分类器通常根据人类生成的文本进行训练，但由于LLM生成的文本偏离其训练数据以及旨在避免检测的对抗性攻击而遭受错误分类。当今的防御策略是被动的，而不是主动的，因为它们依赖于对抗训练或外部检测模型来识别攻击。在这项工作中，我们的目标是识别毒性分类器中导致错误分类的脆弱组件，提出一种基于机械解释性技术的新型策略。我们的研究重点是微调的BERT和RoBERTa分类器，对跨越各种少数群体的不同数据集进行测试。我们使用对抗攻击技术来识别脆弱的电路。最后，我们抑制了这些脆弱的电路，提高了对抗攻击的性能。我们还提供了对这些脆弱电路的人口统计学层面的见解，揭示了模型训练中的公平性和稳健性差距。我们发现模型具有不同的头部，这些头部要么对性能至关重要，要么容易受到攻击，而抑制脆弱的头部可以提高对抗性输入的性能。我们还发现，不同的头部导致了不同人口群体的脆弱性，这可以为毒性检测模型的更具包容性的开发提供信息。



## **9. A Systematic Evaluation of Parameter-Efficient Fine-Tuning Methods for the Security of Code LLMs**

代码LLM安全性的参数高效微调方法的系统评估 cs.CR

25 pages

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12649v1) [paper-pdf](http://arxiv.org/pdf/2509.12649v1)

**Authors**: Kiho Lee, Jungkon Kim, Doowon Kim, Hyoungshick Kim

**Abstract**: Code-generating Large Language Models (LLMs) significantly accelerate software development. However, their frequent generation of insecure code presents serious risks. We present a comprehensive evaluation of seven parameter-efficient fine-tuning (PEFT) techniques, demonstrating substantial gains in secure code generation without compromising functionality. Our research identifies prompt-tuning as the most effective PEFT method, achieving an 80.86% Overall-Secure-Rate on CodeGen2 16B, a 13.5-point improvement over the 67.28% baseline. Optimizing decoding strategies through sampling temperature further elevated security to 87.65%. This equates to a reduction of approximately 203,700 vulnerable code snippets per million generated. Moreover, prompt and prefix tuning increase robustness against poisoning attacks in our TrojanPuzzle evaluation, with strong performance against CWE-79 and CWE-502 attack vectors. Our findings generalize across Python and Java, confirming prompt-tuning's consistent effectiveness. This study provides essential insights and practical guidance for building more resilient software systems with LLMs.

摘要: 代码生成大型语言模型（LLM）显着加速了软件开发。然而，他们频繁生成不安全的代码带来了严重的风险。我们对七种参数高效微调（PEFT）技术进行了全面评估，展示了在不损害功能的情况下在安全代码生成方面的巨大收益。我们的研究将预算调整确定为最有效的PEFT方法，在CodeGen 2 16 B上实现了80.86%的总体安全率，比67.28%的基线提高了13.5个百分点。通过采样温度优化解码策略，安全性进一步提高至87.65%。这相当于每生成的百万个易受攻击的代码片段减少约203，700个。此外，在我们的TrojanPuzzle评估中，提示和前置调整增强了针对中毒攻击的鲁棒性，在针对CWE-79和CWE-502攻击载体的性能强劲。我们的研究结果在Python和Java中得到了推广，证实了预算调优的一致有效性。这项研究为使用LLM构建更具弹性的软件系统提供了重要的见解和实践指导。



## **10. Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs**

LLM联合量化和稀疏化的最佳大脑恢复 cs.CL

Preprint

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11177v2) [paper-pdf](http://arxiv.org/pdf/2509.11177v2)

**Authors**: Hang Guo, Yawei Li, Luca Benini

**Abstract**: Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline.

摘要: 大型语言模型（LLM）压缩的最新进展（例如量化和修剪）取得了显着的成功。然而，随着这些技术逐渐接近各自的极限，依靠单一方法进行进一步压缩变得越来越具有挑战性。在这项工作中，我们通过结合量化和稀疏性来探索替代解决方案。这种联合方法虽然很有希望，但由于对权重分布的固有要求相互冲突，带来了新的困难：量化有利于紧凑的范围，而修剪则受益于高方差。为了解决这个问题，我们提出了最佳大脑恢复（OBR），这是一个通用的、免训练的框架，通过两者之间的误差补偿来协调修剪和量化。OBR通过建立二阶Hessian目标来最大限度地减少下游任务的性能下降，然后通过代理逼近将该目标重新表述为可处理的问题，并最终通过群误差补偿获得封闭形式的解决方案。实验表明，OBR可以在现有LLM上以50%的稀疏度实现积极的W4 A4 KV 4量化，与FP 16密集基线相比，可提供高达4.72倍的加速和6.4倍的内存减少。



## **11. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

EmantSleuth：通过语义意图不变性检测提示注入 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.20890v2) [paper-pdf](http://arxiv.org/pdf/2508.20890v2)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.

摘要: 大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，从虚拟助手到自治代理。然而，它们的灵活性也引入了新的攻击向量，特别是提示注入（PI），其中攻击者通过精心制作的输入操纵模型行为。随着攻击者不断地使用释义、混淆甚至多任务注入策略，现有的基准不再足以捕获所有新兴威胁。   为了解决这一差距，我们构建了一个新的基准，系统地扩展了以前的努力。我们的基准涵盖了两种广泛使用的现有基准，同时引入了新的操纵技术和多任务场景，从而提供了更全面的评估设置。我们发现，现有的防御虽然在原始基准上有效，但在我们的基准下表现出明显的弱点，这凸显了对更强大解决方案的需求。我们的关键见解是，虽然攻击形式可能会有所不同，但对手的意图（注入未经授权的任务）保持不变。在这一观察的基础上，我们提出了EmittSleuth，这是一个面向语义的防御框架，它通过对任务级意图而不是表面特征进行推理来检测提示注入。在最先进的基准测试中进行评估后，AktSleuth始终优于现有的防御，同时保持相当的运行时间和成本效率。这些结果表明，基于意图的语义推理提供了一个强大的，有效的，和可推广的策略，以抵御不断发展的即时注入威胁的LLM。



## **12. Phi: Preference Hijacking in Multi-modal Large Language Models at Inference Time**

Phi：推理时多模式大型语言模型中的偏好劫持 cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.12521v1) [paper-pdf](http://arxiv.org/pdf/2509.12521v1)

**Authors**: Yifan Lan, Yuanpu Cao, Weitong Zhang, Lu Lin, Jinghui Chen

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have gained significant attention across various domains. However, their widespread adoption has also raised serious safety concerns. In this paper, we uncover a new safety risk of MLLMs: the output preference of MLLMs can be arbitrarily manipulated by carefully optimized images. Such attacks often generate contextually relevant yet biased responses that are neither overtly harmful nor unethical, making them difficult to detect. Specifically, we introduce a novel method, Preference Hijacking (Phi), for manipulating the MLLM response preferences using a preference hijacked image. Our method works at inference time and requires no model modifications. Additionally, we introduce a universal hijacking perturbation -- a transferable component that can be embedded into different images to hijack MLLM responses toward any attacker-specified preferences. Experimental results across various tasks demonstrate the effectiveness of our approach. The code for Phi is accessible at https://github.com/Yifan-Lan/Phi.

摘要: 最近，多模式大型语言模型（MLLM）在各个领域引起了密切关注。然而，它们的广泛采用也引发了严重的安全问题。在本文中，我们发现了MLLM的一个新的安全风险：MLLM的输出偏好可以通过精心优化的图像任意操纵。此类攻击通常会产生与上下文相关但有偏见的反应，既不明显有害，也不不道德，因此难以检测。具体来说，我们引入了一种新颖的方法，即偏好劫持（Phi），用于使用偏好劫持的图像来操纵MLLM响应偏好。我们的方法在推理时工作，不需要模型修改。此外，我们还引入了一种通用的劫持扰动--一种可移植的组件，可以嵌入到不同的图像中，以劫持MLLM对任何攻击者指定的偏好的响应。各种任务的实验结果证明了我们方法的有效性。Phi的代码可在https://github.com/Yifan-Lan/Phi上访问。



## **13. Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering**

保持安全！针对问题解答中的间接攻击，对大型语言模型上下文中的安全策略保留进行基准测试 cs.CL

EMNLP 2025 (Main Conference)

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2505.15805v2) [paper-pdf](http://arxiv.org/pdf/2505.15805v2)

**Authors**: Hwan Chang, Yumin Kim, Yonghyun Jun, Hwanhee Lee

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains such as enterprise and government, ensuring that they adhere to user-defined security policies within context is critical-especially with respect to information non-disclosure. While prior LLM studies have focused on general safety and socially sensitive data, large-scale benchmarks for contextual security preservation against attacks remain lacking. To address this, we introduce a novel large-scale benchmark dataset, CoPriva, evaluating LLM adherence to contextual non-disclosure policies in question answering. Derived from realistic contexts, our dataset includes explicit policies and queries designed as direct and challenging indirect attacks seeking prohibited information. We evaluate 10 LLMs on our benchmark and reveal a significant vulnerability: many models violate user-defined policies and leak sensitive information. This failure is particularly severe against indirect attacks, highlighting a critical gap in current LLM safety alignment for sensitive applications. Our analysis reveals that while models can often identify the correct answer to a query, they struggle to incorporate policy constraints during generation. In contrast, they exhibit a partial ability to revise outputs when explicitly prompted. Our findings underscore the urgent need for more robust methods to guarantee contextual security.

摘要: 随着大型语言模型（LLM）越来越多地部署在企业和政府等敏感领域，确保它们在上下文中遵守用户定义的安全策略至关重要，尤其是在信息不披露方面。虽然之前的LLM研究重点关注一般安全和社会敏感数据，但仍然缺乏针对攻击的上下文安全保护的大规模基准。为了解决这个问题，我们引入了一个新颖的大规模基准数据集CoPriva，以评估LLM在问答中对上下文保密政策的遵守情况。我们的数据集源自现实背景，包括明确的政策和查询，旨在作为寻求违禁信息的直接和具有挑战性的间接攻击。我们在我们的基准测试中评估了10个LLM，并揭示了一个重大漏洞：许多模型违反了用户定义的策略并泄漏了敏感信息。这种故障对于间接攻击尤其严重，突出了当前LLM安全对齐敏感应用程序的关键差距。我们的分析表明，虽然模型通常可以识别查询的正确答案，但它们很难在生成过程中纳入政策约束。相比之下，他们表现出部分的能力，修改输出时，明确提示。我们的研究结果强调迫切需要更强大的方法来保证上下文安全。



## **14. Safety Pretraining: Toward the Next Generation of Safe AI**

安全预培训：迈向下一代安全人工智能 cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2504.16980v2) [paper-pdf](http://arxiv.org/pdf/2504.16980v2)

**Authors**: Pratyush Maini, Sachin Goyal, Dylan Sam, Alex Robey, Yash Savani, Yiding Jiang, Andy Zou, Matt Fredrikson, Zacharcy C. Lipton, J. Zico Kolter

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes settings, the risk of generating harmful or toxic content remains a central challenge. Post-hoc alignment methods are brittle: once unsafe patterns are learned during pretraining, they are hard to remove. In this work, we present a data-centric pretraining framework that builds safety into the model from the start. Our framework consists of four key steps: (i) Safety Filtering: building a safety classifier to classify webdata into safe and unsafe categories; (ii) Safety Rephrasing: we recontextualize unsafe webdata into safer narratives; (iii) Native Refusal: we develop RefuseWeb and Moral Education pretraining datasets that actively teach model to refuse on unsafe content and the moral reasoning behind it, and (iv) Harmfulness-Tag annotated pretraining: we flag unsafe content during pretraining using a special token, and use it to steer model away from unsafe generations at inference. Our safety-pretrained models reduce attack success rates from 38.8\% to 8.4\% on standard LLM safety benchmarks with no performance degradation on general tasks.

摘要: 随着大型语言模型（LLM）越来越多地部署在高风险环境中，生成有害或有毒内容的风险仍然是一个核心挑战。事后对齐方法很脆弱：一旦在预训练期间学习到不安全的模式，它们就很难被删除。在这项工作中，我们提出了一个以数据为中心的预训练框架，该框架从一开始就将安全性构建到模型中。我们的框架由四个关键步骤组成：（i）安全过滤：构建一个安全分类器，将网络数据分为安全和不安全类别;（ii）安全改写：我们将不安全的网络数据重新语境化为更安全的叙述;（iii）原生拒绝：我们开发RefuseWeb和道德教育预训练数据集，积极教导模型拒绝不安全内容及其背后的道德推理，和（iv）有害标签注释的预训练：我们在预训练期间使用特殊令牌标记不安全内容，并使用它在推理时引导模型远离不安全的世代。我们的安全预训练模型将标准LLM安全基准的攻击成功率从38.8%降低到8.4%，而一般任务的性能不会下降。



## **15. Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System**

在基于LLM的开发系统中利用工具调用提示实现工具行为劫持 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.05755v2) [paper-pdf](http://arxiv.org/pdf/2509.05755v2)

**Authors**: Yu Liu, Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.

摘要: 基于法学硕士的代理系统利用大型语言模型来处理用户查询、做出决策并执行外部工具，以执行跨聊天机器人、客户服务和软件工程等领域的复杂任务。这些系统的一个关键组件是工具调用提示（TIP），它定义了工具交互协议并指导LLM确保工具使用的安全性和正确性。尽管TIP的安全性很重要，但在很大程度上被忽视了。这项工作调查了与TIP相关的安全风险，揭示了Cursor、Claude Code等基于LLM的主要系统容易受到远程代码执行（RCE）和拒绝服务（NOS）等攻击。通过系统性TIP利用工作流程（TEW），我们通过操纵工具调用演示了外部工具行为劫持。我们还提出了防御机制来增强基于LLM的代理系统中的TIP安全性。



## **16. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

NeuronStrike：对对齐的LLM的神经元级攻击 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11864v1) [paper-pdf](http://arxiv.org/pdf/2509.11864v1)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.

摘要: 安全一致对于大型语言模型（LLM）的道德部署至关重要，指导它们避免生成有害或不道德的内容。当前的对齐技术，例如有监督的微调和来自人类反馈的强化学习，仍然很脆弱，可以被精心设计的对抗提示绕过。不幸的是，此类攻击依赖于试错，缺乏跨模型的通用性，并且受到可扩展性和可靠性的限制。   本文介绍了NeuroStrike，这是一种新颖且可推广的攻击框架，它利用了对齐技术引入的一个基本漏洞：依赖于负责检测和抑制有害输入的稀疏、专门的安全神经元。我们将NeuroStrike应用于白盒和黑盒设置：在白盒设置中，NeuroStrike通过反馈激活分析识别安全神经元，并在推理期间修剪它们以禁用安全机制。在黑匣子环境中，我们提出了第一次LLM剖析攻击，该攻击通过在开权重代理模型上训练对抗提示生成器，然后将它们部署到黑匣子和专有目标上来利用安全神经元的可移植性。我们对来自主要LLM开发商的20多个开量级LLM进行了评估。通过删除目标层中不到0.6%的神经元，NeuroStrike仅使用普通恶意提示即可实现76.9%的平均攻击成功率（ASB）。此外，Neurostrike将四种多模式LLM推广到对不安全图像输入具有100%的ASB。安全神经元在架构之间有效转移，使11个微调模型的ASB达到78.5%，5个提炼模型的ASB达到77.7%。黑匣子LLM分析攻击在包括Google Gemini系列在内的五种黑匣子型号中实现了63.7%的平均ASB。



## **17. One Goal, Many Challenges: Robust Preference Optimization Amid Content-Aware and Multi-Source Noise**

一个目标，诸多挑战：内容感知和多源噪音中的稳健偏好优化 cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2503.12301v2) [paper-pdf](http://arxiv.org/pdf/2503.12301v2)

**Authors**: Amirabbas Afzali, Amirhossein Afsharrad, Seyed Shahabeddin Mousavi, Sanjay Lall

**Abstract**: Large Language Models (LLMs) have made significant strides in generating human-like responses, largely due to preference alignment techniques. However, these methods often assume unbiased human feedback, which is rarely the case in real-world scenarios. This paper introduces Content-Aware Noise-Resilient Preference Optimization (CNRPO), a novel framework that addresses multiple sources of content-dependent noise in preference learning. CNRPO employs a multi-objective optimization approach to separate true preferences from content-aware noises, effectively mitigating their impact. We leverage backdoor attack mechanisms to efficiently learn and control various noise sources within a single model. Theoretical analysis and extensive experiments on different synthetic noisy datasets demonstrate that CNRPO significantly improves alignment with primary human preferences while controlling for secondary noises and biases, such as response length and harmfulness.

摘要: 大型语言模型（LLM）在生成类人响应方面取得了重大进展，这主要归功于偏好对齐技术。然而，这些方法通常假设无偏见的人类反馈，而在现实世界场景中情况很少。本文介绍了内容感知噪音弹性偏好优化（CNRPO），这是一种新型框架，可解决偏好学习中多个内容相关噪音来源。CNRPO采用多目标优化方法将真实偏好与内容感知噪音分开，有效减轻其影响。我们利用后门攻击机制来有效地学习和控制单个模型内的各种噪音源。对不同合成噪音数据集的理论分析和广泛实验表明，CNRPO显着改善了与人类主要偏好的一致性，同时控制次要噪音和偏差，例如响应长度和危害性。



## **18. Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check**

合理的安全调整：通过先检查确保越狱防御 cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11629v1) [paper-pdf](http://arxiv.org/pdf/2509.11629v1)

**Authors**: Chentao Cao, Xiaojun Xu, Bo Han, Hang Li

**Abstract**: As large language models (LLMs) continue to advance in capabilities, ensuring their safety against jailbreak attacks remains a critical challenge. In this paper, we introduce a novel safety alignment approach called Answer-Then-Check, which enhances LLM robustness against malicious prompts by applying thinking ability to mitigate jailbreaking problems before producing a final answer to the user. Our method enables models to directly answer the question in their thought and then critically evaluate its safety before deciding whether to provide it. To implement this approach, we construct the Reasoned Safety Alignment (ReSA) dataset, comprising 80K examples that teach models to reason through direct responses and then analyze their safety. Experimental results demonstrate that our approach achieves the Pareto frontier with superior safety capability while decreasing over-refusal rates on over-refusal benchmarks. Notably, the model fine-tuned with ReSA maintains general reasoning capabilities on benchmarks like MMLU, MATH500, and HumanEval. Besides, our method equips models with the ability to perform safe completion. Unlike post-hoc methods that can only reject harmful queries, our model can provide helpful and safe alternative responses for sensitive topics (e.g., self-harm). Furthermore, we discover that training on a small subset of just 500 examples can achieve comparable performance to using the full dataset, suggesting that safety alignment may require less data than previously assumed.

摘要: 随着大型语言模型（LLM）的能力不断进步，确保其免受越狱攻击的安全性仍然是一个严峻的挑战。在本文中，我们引入了一种名为“Searcher-Then-Check”的新型安全对齐方法，该方法通过在向用户生成最终答案之前应用思维能力来缓解越狱问题，来增强LLM针对恶意提示的鲁棒性。我们的方法使模型能够在思想中直接回答问题，然后在决定是否提供之前批判性地评估其安全性。为了实施这种方法，我们构建了推理安全对齐（ReSA）数据集，其中包括8万个示例，教模型通过直接响应进行推理，然后分析其安全性。实验结果表明，我们的方法以卓越的安全能力达到了帕累托前沿，同时降低了过度拒绝基准上的过度拒绝率。值得注意的是，使用ReSA微调的模型在MMLU，MATH 500和HumanEval等基准上保持了一般推理能力。此外，我们的方法装备模型的能力，执行安全完成。与只能拒绝有害查询的事后方法不同，我们的模型可以为敏感主题（例如，自我伤害）。此外，我们发现，仅对500个示例的一小部分进行训练就可以获得与使用完整数据集相当的性能，这表明安全性对齐可能需要比之前假设的更少的数据。



## **19. Multilingual Collaborative Defense for Large Language Models**

大型语言模型的多语言协作防御 cs.CL

21 pages, 4figures

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2505.11835v2) [paper-pdf](http://arxiv.org/pdf/2505.11835v2)

**Authors**: Hongliang Li, Jinan Xu, Gengping Cui, Changhao Guan, Fengran Mo, Kaiyu Huang

**Abstract**: The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at https://github.com/HLiang-Lee/MCD.

摘要: 大型语言模型（LLM）的稳健性和安全性已成为一个重要的研究领域。一个值得注意的漏洞是，通过将有害查询翻译成罕见或代表性不足的语言来绕过LLM保障措施，这是“越狱”这些模型的一种简单而有效的方法。尽管人们的担忧日益加剧，但针对多语言场景下LLM保护的研究有限，凸显了加强多语言安全的迫切需要。在这项工作中，我们调查了不同语言中的各种攻击特征之间的相关性，并提出了多语言协作防御（MCB），这是一种新型学习方法，可以自动优化连续的软安全提示，以促进LLM的多语言保护。MCB方法具有三个优点：首先，它有效地提高了跨多种语言的性能保护。其次，MCB保持了强大的概括能力，同时最大限度地降低了错误拒绝率。第三，MCB缓解了LLM培训库失衡造成的语言安全失调。为了评估MCB的有效性，我们手动构建常用越狱基准（例如MaliciousDirecct和AdvBench）的多语言版本，以评估各种保障方法。此外，我们以未充分代表（零镜头）语言引入这些数据集，以验证MCB的语言可移植性。结果表明，MCB在防范多语言越狱企图方面优于现有方法，同时还表现出强大的语言传输能力。我们的代码可在https://github.com/HLiang-Lee/MCD上获取。



## **20. Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs**

混乱是最后的障碍：重新思考越狱评估并调查LLM的真正滥用威胁 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2508.16347v2) [paper-pdf](http://arxiv.org/pdf/2508.16347v2)

**Authors**: Yu Yan, Sheng Sun, Zhe Wang, Yijun Lin, Zenghao Duan, zhifei zheng, Min Liu, Zhiyi yin, Jianping Zhang

**Abstract**: With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential.

摘要: 随着大型语言模型（LLM）的发展，许多努力揭示了它们对越狱攻击的脆弱性。尽管这些研究推动了LLM安全调整的进展，但目前尚不清楚LLM是否已经内化了真实的知识来应对现实世界的犯罪，或者只是被迫模拟有毒的语言模式。这种模糊性引发了人们的担忧，即越狱成功通常归因于越狱LLM和法官LLM之间的幻觉循环。通过脱钩越狱技术的使用，我们构建知识密集型问答，以调查LLM在危险知识拥有、有害任务规划效用和有害判断稳健性方面的滥用威胁。实验揭示了LLM的越狱成功率和有害知识拥有之间的不匹配，而现有的LLM作为法官框架往往会将有害判断锚定在有毒语言模式上。我们的研究揭示了现有LLM安全评估与现实世界潜在威胁之间的差距。



## **21. Enhancing Prompt Injection Attacks to LLMs via Poisoning Alignment**

通过中毒对齐增强对LLM的即时注入攻击 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2410.14827v3) [paper-pdf](http://arxiv.org/pdf/2410.14827v3)

**Authors**: Zedian Shao, Hongbin Liu, Jaden Mu, Neil Zhenqiang Gong

**Abstract**: Prompt injection attack, where an attacker injects a prompt into the original one, aiming to make an Large Language Model (LLM) follow the injected prompt to perform an attacker-chosen task, represent a critical security threat. Existing attacks primarily focus on crafting these injections at inference time, treating the LLM itself as a static target. Our experiments show that these attacks achieve some success, but there is still significant room for improvement. In this work, we introduces a more foundational attack vector: poisoning the LLM's alignment process to amplify the success of future prompt injection attacks. Specifically, we propose PoisonedAlign, a method that strategically creates poisoned alignment samples to poison an LLM's alignment dataset. Our experiments across five LLMs and two alignment datasets show that when even a small fraction of the alignment data is poisoned, the resulting model becomes substantially more vulnerable to a wide range of prompt injection attacks. Crucially, this vulnerability is instilled while the LLM's performance on standard capability benchmarks remains largely unchanged, making the manipulation difficult to detect through automated, general-purpose performance evaluations. The code for implementing the attack is available at https://github.com/Sadcardation/PoisonedAlign.

摘要: 提示注入攻击是指攻击者将提示注入到原始提示中，旨在使大型语言模型（LLM）遵循注入的提示来执行攻击者选择的任务，这代表了一种严重的安全威胁。现有的攻击主要集中在推理时制作这些注入，将LLM本身视为静态目标。我们的实验表明，这些攻击取得了一定成功，但仍有很大的改进空间。在这项工作中，我们引入了一个更基本的攻击载体：毒害LLM的对齐过程，以放大未来即时注入攻击的成功。具体来说，我们提出了PoisonedAlign，这是一种策略性地创建有毒比对样本以毒害LLM的比对数据集的方法。我们在五个LLM和两个对齐数据集上进行的实验表明，当即使是一小部分对齐数据被毒害时，生成的模型也会变得更容易受到广泛的即时注入攻击。至关重要的是，该漏洞是在LLM在标准能力基准上的性能基本保持不变的情况下灌输的，从而使得操纵很难通过自动化的通用性能评估检测到。实施攻击的代码可在https://github.com/Sadcardation/PoisonedAlign上获取。



## **22. Securing AI Agents: Implementing Role-Based Access Control for Industrial Applications**

保护人工智能代理：为工业应用程序实施基于角色的访问控制 cs.AI

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11431v1) [paper-pdf](http://arxiv.org/pdf/2509.11431v1)

**Authors**: Aadil Gani Ganie

**Abstract**: The emergence of Large Language Models (LLMs) has significantly advanced solutions across various domains, from political science to software development. However, these models are constrained by their training data, which is static and limited to information available up to a specific date. Additionally, their generalized nature often necessitates fine-tuning -- whether for classification or instructional purposes -- to effectively perform specific downstream tasks. AI agents, leveraging LLMs as their core, mitigate some of these limitations by accessing external tools and real-time data, enabling applications such as live weather reporting and data analysis. In industrial settings, AI agents are transforming operations by enhancing decision-making, predictive maintenance, and process optimization. For example, in manufacturing, AI agents enable near-autonomous systems that boost productivity and support real-time decision-making. Despite these advancements, AI agents remain vulnerable to security threats, including prompt injection attacks, which pose significant risks to their integrity and reliability. To address these challenges, this paper proposes a framework for integrating Role-Based Access Control (RBAC) into AI agents, providing a robust security guardrail. This framework aims to support the effective and scalable deployment of AI agents, with a focus on on-premises implementations.

摘要: 大型语言模型（LLM）的出现为从政治科学到软件开发的各个领域带来了显着的先进解决方案。然而，这些模型受到其训练数据的限制，训练数据是静态的，并且仅限于特定日期之前可用的信息。此外，它们的普遍性通常需要进行微调（无论是出于分类还是教学目的），以有效地执行特定的下游任务。人工智能代理以LLM为核心，通过访问外部工具和实时数据来缓解其中一些限制，从而启用实时天气报告和数据分析等应用程序。在工业环境中，人工智能代理正在通过加强决策、预测性维护和流程优化来改变运营。例如，在制造业中，人工智能代理支持近乎自主的系统，从而提高生产力并支持实时决策。尽管取得了这些进步，人工智能代理仍然容易受到安全威胁，包括即时注入攻击，这对其完整性和可靠性构成了重大风险。为了应对这些挑战，本文提出了一个将基于角色的访问控制（RSC）集成到人工智能代理中的框架，以提供强大的安全护栏。该框架旨在支持人工智能代理的有效且可扩展的部署，重点关注本地实施。



## **23. Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs**

以毒攻毒（F3）：LVLM中一种无需培训且高效的视觉对抗示例净化方法 cs.CV

Accepted by ACM Multimedia 2025 BNI track (Oral)

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.01064v3) [paper-pdf](http://arxiv.org/pdf/2506.01064v3)

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive ``fighting fire with fire'' strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code is available at https://github.com/btzyd/F3.

摘要: 大型视觉语言模型（LVLM）的最新进展展示了它们在广泛的多模式视觉语言任务中的非凡能力。然而，这些模型仍然容易受到视觉对抗攻击，这可能会极大地损害其性能。在本文中，我们介绍了F3，这是一种新型的对抗净化框架，它采用了违反直觉的“以毒攻毒”策略：有意地向对抗性示例引入简单的扰动以减轻其有害影响。具体来说，F3利用从随机干扰的对手示例中获得的跨模式注意力作为参考目标。通过向这些对抗性示例中注入噪音，F3有效地细化了他们的注意力，从而产生更干净、更可靠的模型输出。值得注意的是，这种看似矛盾的利用噪音来抵消对抗攻击的方法产生了令人印象深刻的净化结果。此外，F3具有几个明显的优势：无需训练且易于实施，并且与现有的纯化方法相比，计算效率显着提高。这些属性使F3特别适合大规模工业应用，其中稳健的性能和运营效率都是关键优先事项。该代码可在https://github.com/btzyd/F3上获取。



## **24. Beyond the Protocol: Unveiling Attack Vectors in the Model Context Protocol (MCP) Ecosystem**

超越协议：揭开模型上下文协议（HCP）生态系统中的攻击载体 cs.CR

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.02040v4) [paper-pdf](http://arxiv.org/pdf/2506.02040v4)

**Authors**: Hao Song, Yiming Shen, Wenxuan Luo, Leixin Guo, Ting Chen, Jiashui Wang, Beibei Li, Xiaosong Zhang, Jiachi Chen

**Abstract**: The Model Context Protocol (MCP) is an emerging standard designed to enable seamless interaction between Large Language Model (LLM) applications and external tools or resources. Within a short period, thousands of MCP services have been developed and deployed. However, the client-server integration architecture inherent in MCP may expand the attack surface against LLM Agent systems, introducing new vulnerabilities that allow attackers to exploit by designing malicious MCP servers. In this paper, we present the first end-to-end empirical evaluation of attack vectors targeting the MCP ecosystem. We identify four categories of attacks, i.e., Tool Poisoning Attacks, Puppet Attacks, Rug Pull Attacks, and Exploitation via Malicious External Resources. To evaluate their feasibility, we conduct experiments following the typical steps of launching an attack through malicious MCP servers: upload -> download -> attack. Specifically, we first construct malicious MCP servers and successfully upload them to three widely used MCP aggregation platforms. The results indicate that current audit mechanisms are insufficient to identify and prevent these threats. Next, through a user study and interview with 20 participants, we demonstrate that users struggle to identify malicious MCP servers and often unknowingly install them from aggregator platforms. Finally, we empirically demonstrate that these attacks can trigger harmful actions within the user's local environment, such as accessing private files or controlling devices to transfer digital assets. Additionally, based on interview results, we discuss four key challenges faced by the current MCP security ecosystem. These findings underscore the urgent need for robust security mechanisms to defend against malicious MCP servers and ensure the safe deployment of increasingly autonomous LLM agents.

摘要: 模型上下文协议（HCP）是一种新兴标准，旨在实现大型语言模型（LLM）应用程序与外部工具或资源之间的无缝交互。在短时间内，就开发和部署了数千项HCP服务。然而，LCP固有的客户端-服务器集成架构可能会扩大针对LLM Agent系统的攻击面，引入新的漏洞，允许攻击者通过设计恶意的LCP服务器来利用这些漏洞。本文中，我们首次对针对LCP生态系统的攻击载体进行了端到端的实证评估。我们确定了四类攻击，即工具中毒攻击、木偶攻击、拉地毯攻击以及通过恶意外部资源进行的剥削。为了评估其可行性，我们按照通过恶意LCP服务器发起攻击的典型步骤进行了实验：上传->下载->攻击。具体来说，我们首先构建恶意的LCP服务器，并成功将其上传到三个广泛使用的LCP聚合平台。结果表明，当前的审计机制不足以识别和预防这些威胁。接下来，通过用户研究和对20名参与者的采访，我们证明用户很难识别恶意的LCP服务器，并且通常在不知不觉中从聚合平台安装它们。最后，我们通过经验证明，这些攻击可能会在用户本地环境中引发有害行为，例如访问私人文件或控制设备传输数字资产。此外，根据采访结果，我们讨论了当前HCP安全生态系统面临的四个关键挑战。这些发现凸显了迫切需要强大的安全机制来抵御恶意的HCP服务器，并确保安全部署日益自主的LLM代理。



## **25. Character-Level Perturbations Disrupt LLM Watermarks**

初级扰动扰乱LLM水印 cs.CR

accepted by Network and Distributed System Security (NDSS) Symposium  2026

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.09112v2) [paper-pdf](http://arxiv.org/pdf/2509.09112v2)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.   To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.

摘要: 大型语言模型（LLM）水印将可检测信号嵌入到生成的文本中，以实现版权保护、防止滥用和内容检测。虽然之前的研究使用水印去除攻击来评估稳健性，但这些方法通常不是最优的，从而产生了一种误解，即有效的去除需要大的扰动或强大的对手。   为了弥合差距，我们首先形式化LLM水印的系统模型，并描述了两个受有限访问水印检测器限制的现实威胁模型。然后，我们分析不同类型的扰动在其攻击范围内的变化，即，一次编辑可以影响的代币数量。我们观察到字符级扰动（例如，拼写错误、互换、删除、同字形）可以通过扰乱标记化过程同时影响多个标记。我们证明，在最严格的威胁模型下，字符级扰动对于水印去除来说明显更有效。我们进一步提出了基于遗传算法（GA）的引导删除攻击，使用一个参考检测器进行优化。在一个实际的威胁模型与有限的黑盒查询的水印检测器，我们的方法表现出强大的去除性能。实验证实了字符级扰动的优越性和遗传算法在现实约束条件下去除水印的有效性。此外，我们认为有一个对抗性的困境时，考虑潜在的防御：任何固定的防御可以绕过一个合适的扰动策略。受此原则的启发，我们提出了一种自适应复合字符级攻击。实验结果表明，这种方法可以有效地击败防御。我们的研究结果强调了现有LLM水印方案中的重大漏洞，并强调了开发新的稳健机制的紧迫性。



## **26. Free-MAD: Consensus-Free Multi-Agent Debate**

Free-MAD：无争议的多主体辩论 cs.AI

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11035v1) [paper-pdf](http://arxiv.org/pdf/2509.11035v1)

**Authors**: Yu Cui, Hang Fu, Haibin Zhang, Licheng Wang, Cong Zuo

**Abstract**: Multi-agent debate (MAD) is an emerging approach to improving the reasoning capabilities of large language models (LLMs). Existing MAD methods rely on multiple rounds of interaction among agents to reach consensus, and the final output is selected by majority voting in the last round. However, this consensus-based design faces several limitations. First, multiple rounds of communication increases token overhead and limits scalability. Second, due to the inherent conformity of LLMs, agents that initially produce correct responses may be influenced by incorrect ones during the debate process, causing error propagation. Third, majority voting introduces randomness and unfairness in the decision-making phase, and can degrade the reasoning performance.   To address these issues, we propose \textsc{Free-MAD}, a novel MAD framework that eliminates the need for consensus among agents. \textsc{Free-MAD} introduces a novel score-based decision mechanism that evaluates the entire debate trajectory rather than relying on the last round only. This mechanism tracks how each agent's reasoning evolves, enabling more accurate and fair outcomes. In addition, \textsc{Free-MAD} reconstructs the debate phase by introducing anti-conformity, a mechanism that enables agents to mitigate excessive influence from the majority. Experiments on eight benchmark datasets demonstrate that \textsc{Free-MAD} significantly improves reasoning performance while requiring only a single-round debate and thus reducing token costs. We also show that compared to existing MAD approaches, \textsc{Free-MAD} exhibits improved robustness in real-world attack scenarios.

摘要: 多智能体辩论（MAD）是一种提高大型语言模型（LLM）推理能力的新兴方法。现有的MAD方法依赖于代理之间的多轮交互来达成共识，最终的输出是在最后一轮中通过多数投票选出的。然而，这种基于共识的设计面临着几个限制。首先，多轮通信增加了令牌负担并限制了可扩展性。其次，由于LLM固有的一致性，最初做出正确反应的代理人在辩论过程中可能会受到不正确反应的影响，从而导致错误传播。第三，多数投票在决策阶段引入了随机性和不公平性，并可能降低推理性能。   为了解决这些问题，我们提出了\textsk {Free-MAD}，这是一个新型MAD框架，它消除了代理之间达成共识的需要。\textsk {Free-MAD}引入了一种新颖的基于分数的决策机制，该机制评估整个辩论轨迹，而不是仅依赖最后一轮。该机制跟踪每个代理的推理如何演变，从而实现更准确和公平的结果。此外，\textsk {Free-MAD}通过引入反顺从来重建辩论阶段，反顺从是一种使代理人能够减轻多数人过度影响的机制。对八个基准数据集的实验表明，\textsk {Free-MAD}显着提高了推理性能，同时仅需要单轮辩论，从而降低了代币成本。我们还表明，与现有的MAD方法相比，\textsk {Free-MAD}在现实世界的攻击场景中表现出更好的鲁棒性。



## **27. Public Data Assisted Differentially Private In-Context Learning**

公共数据辅助差异私人背景学习 cs.AI

EMNLP 2025 Findings

**SubmitDate**: 2025-09-13    [abs](http://arxiv.org/abs/2509.10932v1) [paper-pdf](http://arxiv.org/pdf/2509.10932v1)

**Authors**: Seongho Joo, Hyukhun Koh, Kyomin Jung

**Abstract**: In-context learning (ICL) in Large Language Models (LLMs) has shown remarkable performance across various tasks without requiring fine-tuning. However, recent studies have highlighted the risk of private data leakage through the prompt in ICL, especially when LLMs are exposed to malicious attacks. While differential privacy (DP) provides strong privacy guarantees, it often significantly reduces the utility of in-context learning (ICL). To address this challenge, we incorporate task-related public data into the ICL framework while maintaining the DP guarantee. Based on this approach, we propose a private in-context learning algorithm that effectively balances privacy protection and model utility. Through experiments, we demonstrate that our approach significantly improves the utility of private ICL with the assistance of public data. Additionally, we show that our method is robust against membership inference attacks, demonstrating empirical privacy protection.

摘要: 大型语言模型（LLM）中的上下文学习（ICL）在各种任务中表现出出色的性能，无需微调。然而，最近的研究强调了通过ICL中的提示泄露私人数据的风险，特别是当LLM面临恶意攻击时。虽然差异隐私（DP）提供了强有力的隐私保证，但它通常会显着降低上下文学习（ICL）的效用。为了应对这一挑战，我们将与任务相关的公共数据纳入ICL框架，同时保持DP保证。基于这种方法，我们提出了一种私有的上下文学习算法，可以有效平衡隐私保护和模型效用。通过实验，我们证明我们的方法在公共数据的帮助下显着提高了私人ICL的实用性。此外，我们还表明我们的方法对成员资格推断攻击具有鲁棒性，证明了经验性的隐私保护。



## **28. Harmful Prompt Laundering: Jailbreaking LLMs with Abductive Styles and Symbolic Encoding**

有害提示洗钱：具有诱拐风格和象征性编码的越狱LLM cs.AI

EMNLP 2025

**SubmitDate**: 2025-09-13    [abs](http://arxiv.org/abs/2509.10931v1) [paper-pdf](http://arxiv.org/pdf/2509.10931v1)

**Authors**: Seongho Joo, Hyukhun Koh, Kyomin Jung

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their potential misuse for harmful purposes remains a significant concern. To strengthen defenses against such vulnerabilities, it is essential to investigate universal jailbreak attacks that exploit intrinsic weaknesses in the architecture and learning paradigms of LLMs. In response, we propose \textbf{H}armful \textbf{P}rompt \textbf{La}undering (HaPLa), a novel and broadly applicable jailbreaking technique that requires only black-box access to target models. HaPLa incorporates two primary strategies: 1) \textit{abductive framing}, which instructs LLMs to infer plausible intermediate steps toward harmful activities, rather than directly responding to explicit harmful queries; and 2) \textit{symbolic encoding}, a lightweight and flexible approach designed to obfuscate harmful content, given that current LLMs remain sensitive primarily to explicit harmful keywords. Experimental results show that HaPLa achieves over 95% attack success rate on GPT-series models and 70% across all targets. Further analysis with diverse symbolic encoding rules also reveals a fundamental challenge: it remains difficult to safely tune LLMs without significantly diminishing their helpfulness in responding to benign queries.

摘要: 大型语言模型（LLM）在不同的任务中表现出了卓越的能力，但它们可能被滥用于有害目的仍然是一个重大问题。为了加强对这些漏洞的防御，必须调查利用LLM架构和学习范式中固有弱点的通用越狱攻击。作为回应，我们提出了\textbf{H}armful \textbf{P}rompt \textbf{La}undering（HaPLa），这是一种新颖且广泛适用的越狱技术，只需要黑盒访问目标模型。HaPLa包含两个主要策略：1）\textit{溯因框架}，它指示LLM推断出可能的有害活动中间步骤，而不是直接响应明确的有害查询;和2）\textit{符号编码}，一种轻量级和灵活的方法，旨在混淆有害内容，考虑到当前的LLM主要对明确的有害关键字保持敏感。实验结果表明，HaPLa在GPT系列模型上的攻击成功率超过95%，在所有目标上的攻击成功率达到70%。对不同符号编码规则的进一步分析还揭示了一个根本性挑战：在不显着削弱LLM响应良性查询的帮助性的情况下，仍然很难安全地调整LLM。



## **29. LLM in the Middle: A Systematic Review of Threats and Mitigations to Real-World LLM-based Systems**

中间的法学硕士：对现实世界基于法学硕士的系统的威胁和缓解措施的系统性审查 cs.CR

37 pages, 8 figures, 13 tables

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10682v1) [paper-pdf](http://arxiv.org/pdf/2509.10682v1)

**Authors**: Vitor Hugo Galhardo Moia, Igor Jochem Sanz, Gabriel Antonio Fontes Rebello, Rodrigo Duarte de Meneses, Briland Hitaj, Ulf Lindqvist

**Abstract**: The success and wide adoption of generative AI (GenAI), particularly large language models (LLMs), has attracted the attention of cybercriminals seeking to abuse models, steal sensitive data, or disrupt services. Moreover, providing security to LLM-based systems is a great challenge, as both traditional threats to software applications and threats targeting LLMs and their integration must be mitigated. In this survey, we shed light on security and privacy concerns of such LLM-based systems by performing a systematic review and comprehensive categorization of threats and defensive strategies considering the entire software and LLM life cycles. We analyze real-world scenarios with distinct characteristics of LLM usage, spanning from development to operation. In addition, threats are classified according to their severity level and to which scenarios they pertain, facilitating the identification of the most relevant threats. Recommended defense strategies are systematically categorized and mapped to the corresponding life cycle phase and possible attack strategies they attenuate. This work paves the way for consumers and vendors to understand and efficiently mitigate risks during integration of LLMs in their respective solutions or organizations. It also enables the research community to benefit from the discussion of open challenges and edge cases that may hinder the secure and privacy-preserving adoption of LLM-based systems.

摘要: 生成性人工智能（GenAI），特别是大型语言模型（LLM）的成功和广泛采用，引起了寻求滥用模型、窃取敏感数据或破坏服务的网络犯罪分子的注意。此外，为基于LLM的系统提供安全性是一个巨大的挑战，因为必须减轻对软件应用程序的传统威胁和针对LLM及其集成的威胁。在本调查中，我们通过考虑整个软件和LLM生命周期对威胁和防御策略进行系统性审查和全面分类，揭示了此类基于LLM的系统的安全和隐私问题。我们分析具有LLM使用独特特征的现实世界场景，涵盖从开发到运营。此外，威胁还根据其严重程度及其所属场景进行分类，以方便识别最相关的威胁。推荐的防御策略被系统地分类并映射到相应的生命周期阶段和它们削弱的可能攻击策略。这项工作为消费者和供应商了解并有效降低LLM在各自解决方案或组织中集成期间的风险铺平了道路。它还使研究界能够从对可能阻碍安全和隐私保护采用基于LLM的系统的开放挑战和边缘案例的讨论中受益。



## **30. URL2Graph++: Unified Semantic-Structural-Character Learning for Malicious URL Detection**

URL 2Shape ++：用于恶意URL检测的统一语义-结构-字符学习 cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.10287v1) [paper-pdf](http://arxiv.org/pdf/2509.10287v1)

**Authors**: Ye Tian, Yifan Jia, Yanbin Wang, Jianguo Sun, Zhiquan Liu, Xiaowen Ling

**Abstract**: Malicious URL detection remains a major challenge in cybersecurity, primarily due to two factors: (1) the exponential growth of the Internet has led to an immense diversity of URLs, making generalized detection increasingly difficult; and (2) attackers are increasingly employing sophisticated obfuscation techniques to evade detection. We advocate that addressing these challenges fundamentally requires: (1) obtaining semantic understanding to improve generalization across vast and diverse URL sets, and (2) accurately modeling contextual relationships within the structural composition of URLs. In this paper, we propose a novel malicious URL detection method combining multi-granularity graph learning with semantic embedding to jointly capture semantic, character-level, and structural features for robust URL analysis. To model internal dependencies within URLs, we first construct dual-granularity URL graphs at both subword and character levels, where nodes represent URL tokens/characters and edges encode co-occurrence relationships. To obtain fine-grained embeddings, we initialize node representations using a character-level convolutional network. The two graphs are then processed through jointly trained Graph Convolutional Networks to learn consistent graph-level representations, enabling the model to capture complementary structural features that reflect co-occurrence patterns and character-level dependencies. Furthermore, we employ BERT to derive semantic representations of URLs for semantically aware understanding. Finally, we introduce a gated dynamic fusion network to combine the semantically enriched BERT representations with the jointly optimized graph vectors, further enhancing detection performance. We extensively evaluate our method across multiple challenging dimensions. Results show our method exceeds SOTA performance, including against large language models.

摘要: 恶意URL检测仍然是网络安全的一个重大挑战，主要原因是两个因素：（1）互联网的指数级增长导致URL的多样性，使得广义检测变得越来越困难;（2）攻击者越来越多地使用复杂的混淆技术来逃避检测。我们主张解决这些挑战从根本上来说需要：（1）获得语义理解，以改善庞大且多样化的URL集的概括性，（2）准确地建模URL结构组成中的上下文关系。本文提出了一种新型的恶意URL检测方法，该方法将多粒度图学习与语义嵌入相结合，联合捕获语义、字符级和结构特征，以进行稳健的URL分析。为了对URL内的内部依赖关系进行建模，我们首先在子字和字符级别上构建双粒度URL图，其中节点表示URL令牌/字符，边编码共生关系。为了获得细粒度嵌入，我们使用字符级卷积网络初始化节点表示。然后通过联合训练的图卷积网络处理这两个图，以学习一致的图级表示，使模型能够捕获反映共生模式和字符级依赖关系的互补结构特征。此外，我们使用BERT来推导URL的语义表示，以进行语义感知的理解。最后，我们引入门控动态融合网络，将语义丰富的BERT表示与联合优化的图载体相结合，进一步提高检测性能。我们在多个具有挑战性的维度上广泛评估我们的方法。结果表明，我们的方法超过了SOTA性能，包括对大型语言模型。



## **31. Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs**

在岩石和困难之间：利用道德推理越狱法学硕士 cs.CR

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.05367v2) [paper-pdf](http://arxiv.org/pdf/2509.05367v2)

**Authors**: Shei Pern Chua, Zhen Leng Thai, Teh Kai Jun, Xiao Li, Xiaolin Hu

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.

摘要: 大型语言模型（LLM）已经进行了安全调整工作，以减轻有害输出。然而，随着LLM的推理变得更加复杂，他们的智能可能会带来新的安全风险。虽然传统的越狱攻击依赖于单步攻击，但动态适应上下文的多回合越狱策略仍然没有得到充分的研究。在这项工作中，我们引入了TRAL（交互式攻击逻辑的电车问题推理），这是一个利用LLM道德推理来绕过其保障措施的框架。TRAL将对抗目标嵌入以电车问题为模型的道德困境中。TriAL展示了开放和封闭源模型的高越狱成功率。我们的研究结果强调了人工智能安全性的一个根本限制：随着模型获得高级推理能力，它们的对齐性质可能会无意中允许更多隐蔽的安全漏洞被利用。TRAL提出了重新评估安全一致监督策略的迫切需要，因为当前的保障措施可能不足以抵御上下文感知的对抗攻击。



## **32. Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective**

LLM授权的推荐系统的隐私风险：倒置攻击的角度 cs.IR

Accepted at ACM RecSys 2025 (10 pages, 4 figures)

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2508.03703v2) [paper-pdf](http://arxiv.org/pdf/2508.03703v2)

**Authors**: Yubo Wang, Min Tang, Nuo Shen, Shujie Cui, Weiqing Wang

**Abstract**: The large language model (LLM) powered recommendation paradigm has been proposed to address the limitations of traditional recommender systems, which often struggle to handle cold start users or items with new IDs. Despite its effectiveness, this study uncovers that LLM empowered recommender systems are vulnerable to reconstruction attacks that can expose both system and user privacy. To examine this threat, we present the first systematic study on inversion attacks targeting LLM empowered recommender systems, where adversaries attempt to reconstruct original prompts that contain personal preferences, interaction histories, and demographic attributes by exploiting the output logits of recommendation models. We reproduce the vec2text framework and optimize it using our proposed method called Similarity Guided Refinement, enabling more accurate reconstruction of textual prompts from model generated logits. Extensive experiments across two domains (movies and books) and two representative LLM based recommendation models demonstrate that our method achieves high fidelity reconstructions. Specifically, we can recover nearly 65 percent of the user interacted items and correctly infer age and gender in 87 percent of the cases. The experiments also reveal that privacy leakage is largely insensitive to the victim model's performance but highly dependent on domain consistency and prompt complexity. These findings expose critical privacy vulnerabilities in LLM empowered recommender systems.

摘要: 大语言模型（LLM）支持的推荐范式被提出来解决传统推荐系统的局限性，传统推荐系统通常难以处理冷启动用户或具有新ID的项目。尽管有效，这项研究发现LLM授权的推荐系统很容易受到重建攻击，这些攻击可能会暴露系统和用户隐私。为了研究这种威胁，我们对针对LLM授权的推荐系统的倒置攻击进行了首次系统性研究，其中对手试图通过利用推荐模型的输出日志来重建包含个人偏好、交互历史和人口统计属性的原始提示。我们重现vec 2text框架并使用我们提出的名为相似性引导细化的方法对其进行优化，从而能够从模型生成的日志中更准确地重建文本提示。跨两个领域（电影和书籍）的广泛实验和两个代表性的基于LLM的推荐模型证明我们的方法实现了高保真重建。具体来说，我们可以恢复近65%的用户交互项目，并在87%的情况下正确推断年龄和性别。实验还表明，隐私泄露在很大程度上对受害者模型的性能不敏感，但高度依赖于域一致性和提示复杂性。这些发现暴露了LLM授权的推荐系统中的关键隐私漏洞。



## **33. When Your Reviewer is an LLM: Biases, Divergence, and Prompt Injection Risks in Peer Review**

当您的评审员是法学硕士时：同行评审中的偏见、分歧和及时注入风险 cs.CY

**SubmitDate**: 2025-09-12    [abs](http://arxiv.org/abs/2509.09912v1) [paper-pdf](http://arxiv.org/pdf/2509.09912v1)

**Authors**: Changjia Zhu, Junjie Xiong, Renkai Ma, Zhicong Lu, Yao Liu, Lingyao Li

**Abstract**: Peer review is the cornerstone of academic publishing, yet the process is increasingly strained by rising submission volumes, reviewer overload, and expertise mismatches. Large language models (LLMs) are now being used as "reviewer aids," raising concerns about their fairness, consistency, and robustness against indirect prompt injection attacks. This paper presents a systematic evaluation of LLMs as academic reviewers. Using a curated dataset of 1,441 papers from ICLR 2023 and NeurIPS 2022, we evaluate GPT-5-mini against human reviewers across ratings, strengths, and weaknesses. The evaluation employs structured prompting with reference paper calibration, topic modeling, and similarity analysis to compare review content. We further embed covert instructions into PDF submissions to assess LLMs' susceptibility to prompt injection. Our findings show that LLMs consistently inflate ratings for weaker papers while aligning more closely with human judgments on stronger contributions. Moreover, while overarching malicious prompts induce only minor shifts in topical focus, explicitly field-specific instructions successfully manipulate specific aspects of LLM-generated reviews. This study underscores both the promises and perils of integrating LLMs into peer review and points to the importance of designing safeguards that ensure integrity and trust in future review processes.

摘要: 同行评审是学术出版的基石，但由于提交量不断增加、评审员超载和专业知识不匹配，这一过程变得越来越紧张。大型语言模型（LLM）现在被用作“审阅者辅助工具”，这引发了人们对其公平性、一致性和针对间接提示注入攻击的稳健性的担忧。本文对法学硕士作为学术评审员进行了系统评估。使用ICLR 2023和NeurIPS 2022的1，441篇论文的精心策划数据集，我们与人类评论员在评级、优势和劣势方面评估GPT-5-mini。评估采用结构化提示、参考论文校准、主题建模和相似性分析来比较评论内容。我们进一步将秘密指令嵌入到PDF提交中，以评估LLM对即时注射的敏感性。我们的研究结果表明，LLM不断夸大较弱论文的评级，同时更接近人类对较强贡献的判断。此外，虽然总体恶意提示只会导致话题焦点的微小转变，但明确的特定领域指令成功地操纵了LLM生成的评论的特定方面。这项研究强调了将法学硕士纳入同行评审的承诺和危险，并指出了设计确保未来评审流程完整性和信任的保障措施的重要性。



## **34. Advancing Security with Digital Twins: A Comprehensive Survey**

通过数字双胞胎提高安全性：全面调查 cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2505.17310v2) [paper-pdf](http://arxiv.org/pdf/2505.17310v2)

**Authors**: Blessing Airehenbuwa, Touseef Hasan, Souvika Sarkar, Ujjwal Guin

**Abstract**: The proliferation of electronic devices has greatly transformed every aspect of human life, such as communication, healthcare, transportation, and energy. Unfortunately, the global electronics supply chain is vulnerable to various attacks, including piracy of intellectual properties, tampering, counterfeiting, information leakage, side-channel, and fault injection attacks, due to the complex nature of electronic products and vulnerabilities present in them. Although numerous solutions have been proposed to address these threats, significant gaps remain, particularly in providing scalable and comprehensive protection against emerging attacks. Digital twin, a dynamic virtual replica of a physical system, has emerged as a promising solution to address these issues by providing backward traceability, end-to-end visibility, and continuous verification of component integrity and behavior. In this paper, we comprehensively present the latest digital twin-based security implementations, including their role in cyber-physical systems, Internet of Things, cryptographic systems, detection of counterfeit electronics, intrusion detection, fault injection, and side-channel leakage. This work considers these critical security use cases within a single study to offer researchers and practitioners a unified reference for securing hardware with digital twins. The paper also explores the integration of large language models with digital twins for enhanced security and discusses current challenges, solutions, and future research directions.

摘要: 电子设备的激增极大地改变了人类生活的各个方面，例如通信、医疗保健、交通和能源。不幸的是，由于电子产品的复杂性及其存在的漏洞，全球电子供应链很容易受到各种攻击，包括知识产权盗版、篡改、假冒、信息泄露、侧通道和故障注入攻击。尽管已经提出了许多解决方案来解决这些威胁，但仍然存在巨大差距，特别是在针对新出现的攻击提供可扩展和全面的保护方面。Digital twin是物理系统的动态虚拟副本，通过提供向后可追溯性、端到端可见性以及组件完整性和行为的持续验证，已成为解决这些问题的一种有前途的解决方案。在本文中，我们全面介绍了最新的基于数字孪生的安全实现，包括它们在网络物理系统、物联网、加密系统、假冒电子产品检测、入侵检测、故障注入和侧通道泄漏中的作用。这项工作在一项研究中考虑了这些关键的安全用例，为研究人员和从业者提供统一的参考，以保护具有数字双胞胎的硬件。本文还探讨了大型语言模型与数字双胞胎的集成以增强安全性，并讨论了当前的挑战、解决方案和未来的研究方向。



## **35. Steering MoE LLMs via Expert (De)Activation**

通过专家（去）激活MoE LLM cs.CL

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2509.09660v1) [paper-pdf](http://arxiv.org/pdf/2509.09660v1)

**Authors**: Mohsen Fayyaz, Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Ryan Rossi, Trung Bui, Hinrich Schütze, Nanyun Peng

**Abstract**: Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts.

摘要: 大型语言模型（LLM）中的专家混合（MoE）通过专门的前向网络（FFN）子集（称为专家）路由每个令牌。我们介绍了SteerMoE，这是一个通过检测和控制与行为相关的专家来引导MoE模型的框架。我们的检测方法识别出在表现出相反行为的成对输入中具有不同激活模式的专家。通过在推理过程中选择性地（去）激活此类专家，我们可以在无需重新培训或修改权重的情况下控制忠诚和安全等行为。在11个基准和6个LLM中，我们的指导将安全性提高了+20%，忠诚度提高了+27%。在对抗性攻击模式下，它单独会降低-41%，与现有越狱方法结合使用时，安全性会降低-100%，绕过所有安全护栏，暴露了隐藏在专家体内的对齐伪造的新维度。



## **36. Can LLMs Hack Enterprise Networks? Autonomous Assumed Breach Penetration-Testing Active Directory Networks**

LLM可以黑客攻击企业网络吗？自主假设漏洞渗透测试Active目录网络 cs.CR

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2502.04227v3) [paper-pdf](http://arxiv.org/pdf/2502.04227v3)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: Enterprise penetration-testing is often limited by high operational costs and the scarcity of human expertise. This paper investigates the feasibility and effectiveness of using Large Language Model (LLM)-driven autonomous systems to address these challenges in real-world Active Directory (AD) enterprise networks.   We introduce a novel prototype designed to employ LLMs to autonomously perform Assumed Breach penetration-testing against enterprise networks. Our system represents the first demonstration of a fully autonomous, LLM-driven framework capable of compromising accounts within a real-life Microsoft Active Directory testbed, GOAD.   We perform our empirical evaluation using five LLMs, comparing reasoning to non-reasoning models as well as including open-weight models. Through quantitative and qualitative analysis, incorporating insights from cybersecurity experts, we demonstrate that autonomous LLMs can effectively conduct Assumed Breach simulations. Key findings highlight their ability to dynamically adapt attack strategies, perform inter-context attacks (e.g., web-app audits, social engineering, and unstructured data analysis for credentials), and generate scenario-specific attack parameters like realistic password candidates. The prototype exhibits robust self-correction mechanisms, installing missing tools and rectifying invalid command generations.   We find that the associated costs are competitive with, and often significantly lower than, those incurred by professional human pen-testers, suggesting a path toward democratizing access to essential security testing for organizations with budgetary constraints. However, our research also illuminates existing limitations, including instances of LLM ``going down rabbit holes'', challenges in comprehensive information transfer between planning and execution modules, and critical safety concerns that necessitate human oversight.

摘要: 企业渗透测试通常受到高运营成本和人力专业知识稀缺的限制。本文研究了使用大型语言模型（LLM）驱动的自治系统来应对现实世界Active目录（AD）企业网络中这些挑战的可行性和有效性。   我们引入了一种新颖的原型，旨在使用LLM来针对企业网络自主执行假设突破渗透测试。我们的系统首次演示了完全自治的LLM驱动框架，该框架能够在现实生活中的Microsoft Active目录测试平台GOAD中危及帐户。   我们使用五个LLM进行实证评估，将推理与非推理模型进行比较，并包括开放权重模型。通过定量和定性分析，结合网络安全专家的见解，我们证明自治LLM可以有效地进行假设漏洞模拟。主要发现强调了它们动态调整攻击策略、执行上下文间攻击（例如，Web应用程序审计、社会工程和凭证的非结构化数据分析），并生成特定于集群的攻击参数，就像现实的候选密码一样。该原型展示了强大的自我纠正机制，可以安装缺失的工具并纠正无效的命令生成。   我们发现，相关成本与专业人类笔测试人员产生的成本具有竞争力，并且往往远低于专业人类笔测试人员的成本，这为预算限制的组织提供了一条民主化的途径。然而，我们的研究也揭示了现有的局限性，包括LLM“掉进兔子洞”的例子、规划和执行模块之间全面信息传输的挑战，以及需要人工监督的关键安全问题。



## **37. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

IDEATOR：使用自己越狱和基准大型视觉语言模型 cs.CV

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2411.00827v5) [paper-pdf](http://arxiv.org/pdf/2411.00827v5)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLJailbreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.VLJailbreakBench is publicly available at https://roywang021.github.io/VLJailbreakBench.

摘要: 随着大型视觉语言模型（VLM）的日益突出，确保其安全部署变得至关重要。最近的研究探索了VLM针对越狱攻击的鲁棒性--利用模型漏洞来引发有害输出的技术。然而，多样化多模式数据的可用性有限，限制了当前的方法严重依赖于从有害文本数据集派生的对抗性或手动制作的图像，而这些图像通常缺乏跨不同背景的有效性和多样性。本文中，我们提出了IDEATOR，这是一种新型越狱方法，可以自主生成用于黑匣子越狱攻击的恶意图像-文本对。IDEATOR基于这样的见解：VLM本身可以充当强大的红队模型，用于生成多模式越狱提示。具体来说，IDEATOR利用VLM创建有针对性的越狱文本，并将其与由最先进的扩散模型生成的越狱图像配对。大量实验证明了IDEATOR的高效率和可移植性，在越狱MiniGPT-4中平均只需5.34次查询即可实现94%的攻击成功率（ASB），转移到LLaVA、INSTBLIP和Chameleon时，攻击成功率分别为82%、88%和75%。基于IDEATOR强大的可移植性和自动化流程，我们推出了VLJailbreakBench，这是一个由3，654个多模式越狱样本组成的安全基准。我们对最近发布的11个VLM的基准结果揭示了安全一致方面的显着差距。例如，我们的挑战集在GPT-4 o上实现了46.31%的ASB，在Claude-3.5-Sonnet上实现了19.65%的ASB，这凸显了对更强防御的迫切需要。VLJailbreakBench可在https://roywang021.github.io/VLJailbreakBench上公开获取。



## **38. SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models**

SimMark：一种针对大型语言模型的稳健基于句子级相似性的水印算法 cs.CL

Accepted to EMNLP 25 main

**SubmitDate**: 2025-09-11    [abs](http://arxiv.org/abs/2502.02787v2) [paper-pdf](http://arxiv.org/pdf/2502.02787v2)

**Authors**: Amirhossein Dabiriaghdam, Lele Wang

**Abstract**: The widespread adoption of large language models (LLMs) necessitates reliable methods to detect LLM-generated text. We introduce SimMark, a robust sentence-level watermarking algorithm that makes LLMs' outputs traceable without requiring access to model internals, making it compatible with both open and API-based LLMs. By leveraging the similarity of semantic sentence embeddings combined with rejection sampling to embed detectable statistical patterns imperceptible to humans, and employing a soft counting mechanism, SimMark achieves robustness against paraphrasing attacks. Experimental results demonstrate that SimMark sets a new benchmark for robust watermarking of LLM-generated content, surpassing prior sentence-level watermarking techniques in robustness, sampling efficiency, and applicability across diverse domains, all while maintaining the text quality and fluency.

摘要: 大型语言模型（LLM）的广泛采用需要可靠的方法来检测LLM生成的文本。我们引入了SimMark，这是一种强大的业务级水印算法，可以使LLM的输出可追溯，而无需访问模型内部，使其与开放式和基于API的LLM兼容。通过利用语义句子嵌入的相似性与拒绝抽样相结合来嵌入人类无法感知的可检测统计模式，并采用软计数机制，SimMark实现了针对重述攻击的鲁棒性。实验结果表明，SimMark为LLM生成的内容的鲁棒性水印设定了新的基准，在鲁棒性、采样效率和跨不同领域的适用性方面超越了先前的业务级水印技术，同时保持文本质量和流畅性。



## **39. ACE: A Security Architecture for LLM-Integrated App Systems**

ACE：LLM集成应用程序系统的安全架构 cs.CR

25 pages, 13 figures, 8 tables; accepted by Network and Distributed  System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2504.20984v3) [paper-pdf](http://arxiv.org/pdf/2504.20984v3)

**Authors**: Evan Li, Tushin Mallick, Evan Rose, William Robertson, Alina Oprea, Cristina Nita-Rotaru

**Abstract**: LLM-integrated app systems extend the utility of Large Language Models (LLMs) with third-party apps that are invoked by a system LLM using interleaved planning and execution phases to answer user queries. These systems introduce new attack vectors where malicious apps can cause integrity violation of planning or execution, availability breakdown, or privacy compromise during execution.   In this work, we identify new attacks impacting the integrity of planning, as well as the integrity and availability of execution in LLM-integrated apps, and demonstrate them against IsolateGPT, a recent solution designed to mitigate attacks from malicious apps. We propose Abstract-Concrete-Execute (ACE), a new secure architecture for LLM-integrated app systems that provides security guarantees for system planning and execution. Specifically, ACE decouples planning into two phases by first creating an abstract execution plan using only trusted information, and then mapping the abstract plan to a concrete plan using installed system apps. We verify that the plans generated by our system satisfy user-specified secure information flow constraints via static analysis on the structured plan output. During execution, ACE enforces data and capability barriers between apps, and ensures that the execution is conducted according to the trusted abstract plan. We show experimentally that ACE is secure against attacks from the InjecAgent and Agent Security Bench benchmarks for indirect prompt injection, and our newly introduced attacks. We also evaluate the utility of ACE in realistic environments, using the Tool Usage suite from the LangChain benchmark. Our architecture represents a significant advancement towards hardening LLM-based systems using system security principles.

摘要: LLM集成的应用程序系统通过第三方应用程序扩展了大型语言模型（LLM）的实用性，第三方应用程序由系统LLM使用交错的规划和执行阶段调用，以回答用户查询。这些系统引入了新的攻击载体，恶意应用程序可能会导致规划或执行的完整性违反、可用性崩溃或执行期间的隐私受到损害。   在这项工作中，我们识别了影响规划完整性以及LLM集成应用程序中执行完整性和可用性的新攻击，并针对IsolateGPT（旨在减轻恶意应用程序攻击的最新解决方案）进行演示。我们提出Abstract-Concrete-Execute（ACE），这是一种针对LLM集成应用程序系统的新安全架构，为系统规划和执行提供安全保障。具体来说，ACE将规划分为两个阶段，首先仅使用可信信息创建抽象执行计划，然后使用已安装的系统应用程序将抽象计划映射到具体计划。我们通过对结构化计划输出的静态分析来验证系统生成的计划是否满足用户指定的安全信息流约束。在执行过程中，ACE在应用程序之间强制设置数据和能力障碍，并确保执行按照可信的抽象计划进行。我们通过实验表明，ACE可以抵御来自InjectAgent和Agent Security Bench间接即时注入基准的攻击以及我们新引入的攻击。我们还评估了ACE在现实环境中的实用性，使用LangChain基准测试中的工具使用套件。我们的架构代表了使用系统安全原则加强基于LLM的系统的重大进步。



## **40. Architecting Resilient LLM Agents: A Guide to Secure Plan-then-Execute Implementations**

构建弹性LLM代理：安全计划后执行实施指南 cs.CR

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08646v1) [paper-pdf](http://arxiv.org/pdf/2509.08646v1)

**Authors**: Ron F. Del Rosario, Klaudia Krawiecka, Christian Schroeder de Witt

**Abstract**: As Large Language Model (LLM) agents become increasingly capable of automating complex, multi-step tasks, the need for robust, secure, and predictable architectural patterns is paramount. This paper provides a comprehensive guide to the ``Plan-then-Execute'' (P-t-E) pattern, an agentic design that separates strategic planning from tactical execution. We explore the foundational principles of P-t-E, detailing its core components - the Planner and the Executor - and its architectural advantages in predictability, cost-efficiency, and reasoning quality over reactive patterns like ReAct (Reason + Act). A central focus is placed on the security implications of this design, particularly its inherent resilience to indirect prompt injection attacks by establishing control-flow integrity. We argue that while P-t-E provides a strong foundation, a defense-in-depth strategy is necessary, and we detail essential complementary controls such as the Principle of Least Privilege, task-scoped tool access, and sandboxed code execution. To make these principles actionable, this guide provides detailed implementation blueprints and working code references for three leading agentic frameworks: LangChain (via LangGraph), CrewAI, and AutoGen. Each framework's approach to implementing the P-t-E pattern is analyzed, highlighting unique features like LangGraph's stateful graphs for re-planning, CrewAI's declarative tool scoping for security, and AutoGen's built-in Docker sandboxing. Finally, we discuss advanced patterns, including dynamic re-planning loops, parallel execution with Directed Acyclic Graphs (DAGs), and the critical role of Human-in-the-Loop (HITL) verification, to offer a complete strategic blueprint for architects, developers, and security engineers aiming to build production-grade, resilient, and trustworthy LLM agents.

摘要: 随着大型语言模型（LLM）代理越来越有能力自动化复杂的多步骤任务，对稳健、安全且可预测的架构模式的需求变得至关重要。本文提供了“计划然后执行”（P-t-E）模式的全面指南，这是一种将战略规划与战术执行分开的代理设计。我们探索了P-t-E的基本原则，详细介绍了其核心组件--计划者和执行者--以及与ReAct（Reason + Act）等反应式模式相比，其在可预测性、成本效率和推理质量方面的架构优势。中心焦点是该设计的安全影响，特别是其通过建立控制流完整性来对间接即时注入攻击的固有弹性。我们认为，虽然P-t-E提供了坚实的基础，但深度防御策略是必要的，并且我们详细介绍了必要的补充控制，例如最小特权原则、任务范围的工具访问和沙箱代码执行。为了使这些原则具有可操作性，本指南为三个领先的代理框架提供了详细的实现蓝图和工作代码参考：LangChain（通过LangCurve）、CrewAI和AutoGen。分析了每个框架实现P-t-E模式的方法，重点介绍了Langstra用于重新规划的有状态图、CrewAI用于安全的声明性工具范围以及AutoGen的内置Docker沙箱等独特功能。最后，我们讨论了高级模式，包括动态重新规划循环、与有向无环图（DAB）的并行执行以及人在环（HITL）验证的关键作用，为旨在构建生产级、弹性且值得信赖的LLM代理的架构师、开发人员和安全工程师提供完整的战略蓝图。



## **41. Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors**

您的语言模型可以像人类一样秘密写作：对LLM生成的文本检测器的对比重述攻击 cs.CL

Accepted by EMNLP-2025

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2505.15337v3) [paper-pdf](http://arxiv.org/pdf/2505.15337v3)

**Authors**: Hao Fang, Jiawei Kong, Tianqu Zhuang, Yixiang Qiu, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Abstract**: The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.

摘要: 学术抄袭等大型语言模型（LLM）的滥用推动了识别LLM生成文本的检测器的发展。为了绕过这些检测器，出现了故意重写这些文本以逃避检测的重述攻击。尽管取得了成功，但现有方法需要大量的数据和计算预算来训练专门的解释器，并且当面对先进的检测算法时，它们的攻击功效会大大降低。为了解决这个问题，我们提出了\textBF{Co} ntrasive\textBF{P}araphrase \textBF{A}ttack（CoPA），这是一种免训练方法，可以使用现成的LLM有效地欺骗文本检测器。第一步是仔细编写指令，鼓励LLM生成更多类似人类的文本。尽管如此，我们观察到LLM固有的统计偏差仍然会导致一些生成的文本携带某些可以被检测器捕获的类似机器的属性。为了克服这个问题，CoPA构建了一个辅助的类似机器的单词分布，与LLM生成的类似人类的分布形成对比。通过在解码过程中从类人分布中减去类机器模式，CoPA能够生成文本检测器难以识别的句子。我们的理论分析表明了拟议攻击的优越性。大量实验验证了CoPA在各种场景中欺骗文本检测器的有效性。



## **42. TraceRAG: A LLM-Based Framework for Explainable Android Malware Detection and Behavior Analysis**

TraceRAG：一个基于LLM的可解释Android恶意软件检测和行为分析框架 cs.SE

**SubmitDate**: 2025-09-10    [abs](http://arxiv.org/abs/2509.08865v1) [paper-pdf](http://arxiv.org/pdf/2509.08865v1)

**Authors**: Guangyu Zhang, Xixuan Wang, Shiyu Sun, Peiyan Xiao, Kun Sun, Yanhai Xiong

**Abstract**: Sophisticated evasion tactics in malicious Android applications, combined with their intricate behavioral semantics, enable attackers to conceal malicious logic within legitimate functions, underscoring the critical need for robust and in-depth analysis frameworks. However, traditional analysis techniques often fail to recover deeply hidden behaviors or provide human-readable justifications for their decisions. Inspired by advances in large language models (LLMs), we introduce TraceRAG, a retrieval-augmented generation (RAG) framework that bridges natural language queries and Java code to deliver explainable malware detection and analysis. First, TraceRAG generates summaries of method-level code snippets, which are indexed in a vector database. At query time, behavior-focused questions retrieve the most semantically relevant snippets for deeper inspection. Finally, based on the multi-turn analysis results, TraceRAG produces human-readable reports that present the identified malicious behaviors and their corresponding code implementations. Experimental results demonstrate that our method achieves 96\% malware detection accuracy and 83.81\% behavior identification accuracy based on updated VirusTotal (VT) scans and manual verification. Furthermore, expert evaluation confirms the practical utility of the reports generated by TraceRAG.

摘要: 恶意Android应用程序中复杂的规避策略，再加上复杂的行为语义，使攻击者能够将恶意逻辑隐藏在合法功能中，凸显了对稳健和深入分析框架的迫切需求。然而，传统的分析技术往往无法恢复隐藏的行为，也无法为其决策提供人类可读的理由。受大型语言模型（LLM）进步的启发，我们引入了TraceRAG，这是一种检索增强生成（RAG）框架，它将自然语言查询和Java代码连接起来，以提供可解释的恶意软件检测和分析。首先，TraceRAG生成方法级代码片段的摘要，这些代码片段在载体数据库中编入索引。在查询时，以行为为中心的问题检索最相关的语义片段以进行更深入的检查。最后，根据多轮分析结果，TraceRAG生成人类可读的报告，其中呈现识别的恶意行为及其相应的代码实现。实验结果表明，基于更新的Virus Total（VT）扫描和手动验证，我们的方法实现了96%的恶意软件检测准确率和83.81%的行为识别准确率。此外，专家评估证实了TraceRAG生成的报告的实际用途。



## **43. ImportSnare: Directed "Code Manual" Hijacking in Retrieval-Augmented Code Generation**

ImportSnare：检索增强代码生成中的定向“代码手册”劫持 cs.CR

This paper has been accepted by the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07941v1) [paper-pdf](http://arxiv.org/pdf/2509.07941v1)

**Authors**: Kai Ye, Liangcai Su, Chenxiong Qian

**Abstract**: Code generation has emerged as a pivotal capability of Large Language Models(LLMs), revolutionizing development efficiency for programmers of all skill levels. However, the complexity of data structures and algorithmic logic often results in functional deficiencies and security vulnerabilities in generated code, reducing it to a prototype requiring extensive manual debugging. While Retrieval-Augmented Generation (RAG) can enhance correctness and security by leveraging external code manuals, it simultaneously introduces new attack surfaces.   In this paper, we pioneer the exploration of attack surfaces in Retrieval-Augmented Code Generation (RACG), focusing on malicious dependency hijacking. We demonstrate how poisoned documentation containing hidden malicious dependencies (e.g., matplotlib_safe) can subvert RACG, exploiting dual trust chains: LLM reliance on RAG and developers' blind trust in LLM suggestions. To construct poisoned documents, we propose ImportSnare, a novel attack framework employing two synergistic strategies: 1)Position-aware beam search optimizes hidden ranking sequences to elevate poisoned documents in retrieval results, and 2)Multilingual inductive suggestions generate jailbreaking sequences to manipulate LLMs into recommending malicious dependencies. Through extensive experiments across Python, Rust, and JavaScript, ImportSnare achieves significant attack success rates (over 50% for popular libraries such as matplotlib and seaborn) in general, and is also able to succeed even when the poisoning ratio is as low as 0.01%, targeting both custom and real-world malicious packages. Our findings reveal critical supply chain risks in LLM-powered development, highlighting inadequate security alignment for code generation tasks. To support future research, we will release the multilingual benchmark suite and datasets. The project homepage is https://importsnare.github.io.

摘要: 代码生成已成为大型语言模型（LLM）的关键功能，彻底改变了所有技能水平的程序员的开发效率。然而，数据结构和算法逻辑的复杂性往往会导致生成的代码中存在功能缺陷和安全漏洞，从而将其简化为需要大量手动调试的原型。虽然检索增强生成（RAG）可以通过利用外部代码手册来增强正确性和安全性，但它同时引入了新的攻击表面。   在本文中，我们率先探索检索增强代码生成（RACG）中的攻击表面，重点关注恶意依赖劫持。我们演示了如何包含隐藏恶意依赖项的有毒文档（例如，matplotlib_safe）可以利用双重信任链颠覆RACG：LLM对RAG的依赖以及开发人员对LLM建议的盲目信任。为了构建有毒文档，我们提出了ImportSnare，这是一种采用两种协同策略的新型攻击框架：1）位置感知射束搜索优化隐藏排名序列，以提升检索结果中的有毒文档，2）多语言归纳建议生成越狱序列，以操纵LLM推荐恶意依赖项。通过Python、Rust和JavaScript的广泛实验，ImportSnare总体上实现了显着的攻击成功率（matplotlib和seaborn等流行库超过50%），并且即使中毒率低至0.01%，也能够成功，针对自定义和现实世界的恶意软件包。我们的研究结果揭示了LLM驱动的开发中的关键供应链风险，凸显了代码生成任务的安全一致性不足。为了支持未来的研究，我们将发布多语言基准套件和数据集。项目主页为https://importsnare.github.io。



## **44. Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees**

使用结构化攻击树的LLM驱动渗透测试中的引导推理 cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07939v1) [paper-pdf](http://arxiv.org/pdf/2509.07939v1)

**Authors**: Katsuaki Nakano, Reza Feyyazi, Shanchieh Jay Yang, Michael Zuzak

**Abstract**: Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments

摘要: 大型语言模型（LLM）的最新进展激发了人们对自动化网络安全渗透测试工作流程的兴趣，为企业系统提供更快、更一致的漏洞评估。现有的用于渗透测试的LLM代理主要依赖于自我引导推理，这可能会产生不准确或幻觉的程序步骤。因此，LLM代理可能会采取非生产性的行动，例如利用未使用的软件库或生成重复先前策略的周期性响应。在这项工作中，我们提出了一个用于渗透测试LLM代理的引导推理管道，该管道结合了从MITRE ATT & CK矩阵（一个经过验证的渗透测试kll链）构建的确定性任务树，以将LLM的重组过程限制为明确定义的策略、技术和程序。这将推理锚定在经过验证的渗透测试方法中，并通过引导代理走向更有成效的攻击程序来过滤无效操作。为了评估我们的方法，我们使用三个LLM（Llama-3-8B、Gemini-1.5和GPT-4）构建了一个自动渗透测试LLM代理，并将其应用于导航10个HackTheBox网络安全演习，其中包含103个代表现实世界网络攻击场景的离散子任务。我们提出的推理管道使用Llama-3-8B、Gemini-1.5和GPT-4分别引导LLM代理完成71.8%、72.8%和78.6%的子任务。相比之下，使用自我引导推理的最先进的LLM渗透测试工具仅完成了13.5%、16.5%和75.7%的子任务，并且需要86.2%、118.7%和205.9%的模型查询。这表明将确定性任务树纳入LLM推理管道可以提高自动化网络安全评估的准确性和效率



## **45. AgentSentinel: An End-to-End and Real-Time Security Defense Framework for Computer-Use Agents**

AgentSentinel：计算机使用代理的端到端实时安全防御框架 cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07764v1) [paper-pdf](http://arxiv.org/pdf/2509.07764v1)

**Authors**: Haitao Hu, Peng Chen, Yanpeng Zhao, Yuqi Chen

**Abstract**: Large Language Models (LLMs) have been increasingly integrated into computer-use agents, which can autonomously operate tools on a user's computer to accomplish complex tasks. However, due to the inherently unstable and unpredictable nature of LLM outputs, they may issue unintended tool commands or incorrect inputs, leading to potentially harmful operations. Unlike traditional security risks stemming from insecure user prompts, tool execution results from LLM-driven decisions introduce new and unique security challenges. These vulnerabilities span across all components of a computer-use agent. To mitigate these risks, we propose AgentSentinel, an end-to-end, real-time defense framework designed to mitigate potential security threats on a user's computer. AgentSentinel intercepts all sensitive operations within agent-related services and halts execution until a comprehensive security audit is completed. Our security auditing mechanism introduces a novel inspection process that correlates the current task context with system traces generated during task execution. To thoroughly evaluate AgentSentinel, we present BadComputerUse, a benchmark consisting of 60 diverse attack scenarios across six attack categories. The benchmark demonstrates a 87% average attack success rate on four state-of-the-art LLMs. Our evaluation shows that AgentSentinel achieves an average defense success rate of 79.6%, significantly outperforming all baseline defenses.

摘要: 大型语言模型（LLM）已越来越多地集成到计算机使用代理中，这些代理可以在用户计算机上自主操作工具来完成复杂的任务。然而，由于LLM输出本质上不稳定和不可预测，它们可能会发出意外的工具命令或不正确的输入，从而导致潜在的有害操作。与由不安全的用户提示产生的传统安全风险不同，LLM驱动的决策产生的工具执行会带来新的独特安全挑战。这些漏洞跨越计算机使用代理的所有组件。为了减轻这些风险，我们提出了AgentSentinel，这是一种端到端的实时防御框架，旨在减轻用户计算机上的潜在安全威胁。AgentSentinel拦截代理相关服务中的所有敏感操作，并停止执行，直到完成全面的安全审计。我们的安全审计机制引入了一种新颖的检查过程，该过程将当前任务上下文与任务执行期间生成的系统跟踪关联起来。为了彻底评估AgentSentinel，我们介绍了BadComputerUse，这是一个由六种攻击类别的60种不同攻击场景组成的基准。该基准显示，四种最先进的LLM的平均攻击成功率为87%。我们的评估显示，AgentSentinel的平均防御成功率为79.6%，显着优于所有基线防御。



## **46. Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling**

通过激活引导MCMC采样的可转移直接即时注射 cs.AI

Accepted to EMNLP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07617v1) [paper-pdf](http://arxiv.org/pdf/2509.07617v1)

**Authors**: Minghui Li, Hao Zhang, Yechao Zhang, Wei Wan, Shengshan Hu, pei Xiaobing, Jing Wang

**Abstract**: Direct Prompt Injection (DPI) attacks pose a critical security threat to Large Language Models (LLMs) due to their low barrier of execution and high potential damage. To address the impracticality of existing white-box/gray-box methods and the poor transferability of black-box methods, we propose an activations-guided prompt injection attack framework. We first construct an Energy-based Model (EBM) using activations from a surrogate model to evaluate the quality of adversarial prompts. Guided by the trained EBM, we employ the token-level Markov Chain Monte Carlo (MCMC) sampling to adaptively optimize adversarial prompts, thereby enabling gradient-free black-box attacks. Experimental results demonstrate our superior cross-model transferability, achieving 49.6% attack success rate (ASR) across five mainstream LLMs and 34.6% improvement over human-crafted prompts, and maintaining 36.6% ASR on unseen task scenarios. Interpretability analysis reveals a correlation between activations and attack effectiveness, highlighting the critical role of semantic patterns in transferable vulnerability exploitation.

摘要: 直接提示注入（DPI）攻击由于其低执行门槛和高潜在危害性，对大型语言模型（LLM）构成了严重的安全威胁。针对现有白盒/灰盒方法的不实用性和黑盒方法的可移植性差的问题，提出了一种激活引导的提示注入攻击框架.首先，我们构建了一个基于能量的模型（EBM）使用代理模型的激活来评估对抗性提示的质量。在经过训练的EBM的指导下，我们采用令牌级马尔可夫链蒙特卡罗（MCMC）采样来自适应地优化对抗性提示，从而实现无梯度的黑盒攻击。实验结果证明了我们卓越的跨模型可移植性，在五种主流LLM中实现了49.6%的攻击成功率（ASB），比人工制作的提示提高了34.6%，并在未见的任务场景中保持了36.6%的ASB。可解释性分析揭示了激活和攻击有效性之间的相关性，凸显了语义模式在可转移漏洞利用中的关键作用。



## **47. A Decade-long Landscape of Advanced Persistent Threats: Longitudinal Analysis and Global Trends**

长达十年的高级持续威胁格局：纵向分析和全球趋势 cs.CR

18 pages, 13 figures (including subfigures), 11 tables. To appear in  the Proceedings of the ACM Conference on Computer and Communications Security  (CCS) 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07457v1) [paper-pdf](http://arxiv.org/pdf/2509.07457v1)

**Authors**: Shakhzod Yuldoshkhujaev, Mijin Jeon, Doowon Kim, Nick Nikiforakis, Hyungjoon Koo

**Abstract**: An advanced persistent threat (APT) refers to a covert, long-term cyberattack, typically conducted by state-sponsored actors, targeting critical sectors and often remaining undetected for long periods. In response, collective intelligence from around the globe collaborates to identify and trace surreptitious activities, generating substantial documentation on APT campaigns publicly available on the web. While prior works predominantly focus on specific aspects of APT cases, such as detection, evaluation, cyber threat intelligence, and dataset creation, limited attention has been devoted to revisiting and investigating these scattered dossiers in a longitudinal manner. The objective of our study is to fill the gap by offering a macro perspective, connecting key insights and global trends in past APT attacks. We systematically analyze six reliable sources-three focused on technical reports and another three on threat actors-examining 1,509 APT dossiers (24,215 pages) spanning 2014-2023, and identifying 603 unique APT groups worldwide. To efficiently unearth relevant information, we employ a hybrid methodology that combines rule-based information retrieval with large-language-model-based search techniques. Our longitudinal analysis reveals shifts in threat actor activities, global attack vectors, changes in targeted sectors, and relationships between cyberattacks and significant events such as elections or wars, which provide insights into historical patterns in APT evolution. Over the past decade, 154 countries have been affected, primarily using malicious documents and spear phishing as dominant initial infiltration vectors, with a noticeable decline in zero-day exploitation since 2016. Furthermore, we present our findings through interactive visualization tools, such as an APT map or flow diagram, to facilitate intuitive understanding of global patterns and trends in APT activities.

摘要: 高级持续威胁（APT）是指秘密的长期网络攻击，通常由国家支持的行为者实施，针对关键部门，并且通常长时间未被发现。作为回应，来自全球各地的集体情报机构合作识别和追踪秘密活动，在网上公开生成有关APT活动的大量文档。虽然之前的工作主要集中在APT案例的特定方面，例如检测、评估、网络威胁情报和数据集创建，但人们对以纵向方式重新访问和调查这些分散的档案的注意力有限。我们研究的目标是通过提供宏观视角、联系过去APT攻击的关键见解和全球趋势来填补这一空白。我们系统地分析了六个可靠的来源，其中三个集中在技术报告上，另外三个集中在威胁行为者上，检查了2014-2023年的1，509份APT档案（24，215页），并确定了全球603个独特的APT组织。为了有效地挖掘相关信息，我们采用了一种混合的方法，结合基于规则的信息检索与基于大语言模型的搜索技术。我们的纵向分析揭示了威胁行为者活动的变化、全球攻击载体、目标行业的变化以及网络攻击与选举或战争等重大事件之间的关系，从而深入了解APT演变的历史模式。在过去的十年中，154个国家受到影响，主要使用恶意文件和鱼叉式网络钓鱼作为主要的初始渗透载体，自2016年以来零日攻击明显下降。此外，我们还通过交互式可视化工具（例如APT地图或流程图）展示我们的发现，以促进对APT活动的全球模式和趋势的直观理解。



## **48. OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models**

Obliviate：针对大型语言模型的稳健且实用的机器去学习 cs.CL

To appear at EMNLP 25 main conference

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.04416v2) [paper-pdf](http://arxiv.org/pdf/2505.04416v2)

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose \textbf{OBLIVIATE}, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA) ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: \emph{forget quality} (via a new document-level memorization score), \emph{model utility}, and \emph{fluency}. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios.

摘要: 在广泛的库中训练的大型语言模型（LLM）存在记忆敏感、受版权保护或有毒内容的风险。为了解决这个问题，我们提出了\textBF{ObLIVIATE}，这是一个强大的去学习框架，可以在保留模型效用的同时删除目标数据。该框架遵循一个结构化过程：提取目标令牌、构建保留集以及使用定制的损失函数进行微调，该函数包括三个部分--掩蔽、蒸馏和世界事实。使用低级适配器（LoRA）可以在不影响取消学习质量的情况下确保效率。我们使用一套全面的指标对多个数据集进行实验，包括《哈利·波特》系列、WMDP和TOFU：\{忘记质量}（通过新的文档级记忆评分）、\{模型实用程序}和\{流利度}。结果证明了它在抵抗隶属度推理攻击、最大限度地减少对保留数据的影响以及在不同场景下保持稳健性方面的有效性。



## **49. GRADA: Graph-based Reranking against Adversarial Documents Attack**

GRADA：基于图的重新排名对抗文档攻击 cs.IR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.07546v2) [paper-pdf](http://arxiv.org/pdf/2505.07546v2)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.

摘要: 检索增强生成（RAG）框架通过集成来自检索文档的外部知识来提高大型语言模型（LLM）的准确性，从而克服模型静态内在知识的局限性。然而，这些系统很容易受到对抗性攻击，这些攻击通过引入对抗性但在语义上与查询相似的文档来操纵检索过程。值得注意的是，虽然这些对抗性文档类似于查询，但它们与检索集中的良性文档表现出弱的相似性。因此，我们提出了一个简单而有效的基于图形的对抗性文档攻击重新排名（GRADA）框架，旨在保留检索质量，同时显着降低对手的成功。我们的研究通过在五个LLM上进行的实验来评估我们的方法的有效性：GPT-3.5-Turbo，GPT-4 o，Llama3.1-8b，Llama3.1- 70 b和Qwen2.5- 7 b。我们使用三个数据集来评估性能，来自Natural Questions数据集的结果表明攻击成功率降低了80%，同时保持了最小的准确性损失。



## **50. Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection**

多轮会话中的社会工程个性化攻击：LLM Agent仿真与检测 cs.CR

Accepted as a paper at COLM 2025 Workshop on AI Agents: Capabilities  and Safety

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.15552v2) [paper-pdf](http://arxiv.org/pdf/2503.15552v2)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the SE attack mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.

摘要: 会话代理的快速发展，特别是由大型语言模型（LLM）驱动的聊天机器人，对社交媒体平台构成了社会工程（SE）攻击的重大风险。由于这些会话的动态性质，多回合、基于聊天的交互中的SE检测比单实例检测复杂得多。减轻这种威胁的一个关键因素是了解SE攻击的机制，特别是攻击者如何利用漏洞以及受害者的个性特征如何影响他们的易感性。在这项工作中，我们提出了一个LLM-agentic框架，SE-VSim，模拟SE攻击机制，通过生成多轮对话。我们对具有不同性格特征的受害者特工进行建模，以评估心理特征如何影响操纵的易感性。我们使用包含1000多个模拟对话的数据集，检查了冒充招聘人员、资助机构和记者的对手试图提取敏感信息的攻击场景。基于此分析，我们提出了一个概念验证SE-OmniGuard，通过利用受害者个性的先验知识、评估攻击策略以及监控对话中的信息交换以识别潜在的SE尝试，为用户提供个性化保护。




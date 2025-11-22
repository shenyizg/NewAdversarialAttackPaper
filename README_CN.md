# Latest Adversarial Attack Papers
**update at 2025-11-22 11:18:45**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Large Language Model-Based Reward Design for Deep Reinforcement Learning-Driven Autonomous Cyber Defense**

基于大语言模型的深度强化学习驱动的自主网络防御奖励设计 cs.LG

Accepted in the AAAI-26 Workshop on Artificial Intelligence for Cyber Security (AICS)

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16483v1) [paper-pdf](https://arxiv.org/pdf/2511.16483v1)

**Authors**: Sayak Mukherjee, Samrat Chatterjee, Emilie Purvine, Ted Fujimoto, Tegan Emerson

**Abstract**: Designing rewards for autonomous cyber attack and defense learning agents in a complex, dynamic environment is a challenging task for subject matter experts. We propose a large language model (LLM)-based reward design approach to generate autonomous cyber defense policies in a deep reinforcement learning (DRL)-driven experimental simulation environment. Multiple attack and defense agent personas were crafted, reflecting heterogeneity in agent actions, to generate LLM-guided reward designs where the LLM was first provided with contextual cyber simulation environment information. These reward structures were then utilized within a DRL-driven attack-defense simulation environment to learn an ensemble of cyber defense policies. Our results suggest that LLM-guided reward designs can lead to effective defense strategies against diverse adversarial behaviors.

摘要: 对于主题专家来说，在复杂、动态的环境中为自主网络攻击和防御学习代理设计奖励是一项具有挑战性的任务。我们提出了一种基于大语言模型（LLM）的奖励设计方法，以在深度强化学习（DRL）驱动的实验模拟环境中生成自主网络防御策略。精心设计了多个攻击和防御代理角色，反映了代理动作的多样性，以生成LLM引导的奖励设计，其中LLM首先被提供上下文网络模拟环境信息。然后在DRL驱动的攻击防御模拟环境中使用这些奖励结构来学习一整套网络防御策略。我们的结果表明，LLM指导的奖励设计可以制定针对不同对抗行为的有效防御策略。



## **2. Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security**

Q-MLLM：鲁棒多模式大型语言模型安全性的载体量化 cs.CR

Accepted by NDSS 2026

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16229v1) [paper-pdf](https://arxiv.org/pdf/2511.16229v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in cross-modal understanding, but remain vulnerable to adversarial attacks through visual inputs despite robust textual safety mechanisms. These vulnerabilities arise from two core weaknesses: the continuous nature of visual representations, which allows for gradient-based attacks, and the inadequate transfer of text-based safety mechanisms to visual content. We introduce Q-MLLM, a novel architecture that integrates two-level vector quantization to create a discrete bottleneck against adversarial attacks while preserving multimodal reasoning capabilities. By discretizing visual representations at both pixel-patch and semantic levels, Q-MLLM blocks attack pathways and bridges the cross-modal safety alignment gap. Our two-stage training methodology ensures robust learning while maintaining model utility. Experiments demonstrate that Q-MLLM achieves significantly better defense success rate against both jailbreak attacks and toxic image attacks than existing approaches. Notably, Q-MLLM achieves perfect defense success rate (100\%) against jailbreak attacks except in one arguable case, while maintaining competitive performance on multiple utility benchmarks with minimal inference overhead. This work establishes vector quantization as an effective defense mechanism for secure multimodal AI systems without requiring expensive safety-specific fine-tuning or detection overhead. Code is available at https://github.com/Amadeuszhao/QMLLM.

摘要: 多模式大型语言模型（MLLM）在跨模式理解方面表现出了令人印象深刻的能力，但尽管具有强大的文本安全机制，但仍然容易受到视觉输入的对抗攻击。这些漏洞源于两个核心弱点：视觉表示的连续性（允许基于梯度的攻击）以及基于文本的安全机制向视觉内容的不充分转移。我们引入了Q-MLLM，这是一种新颖的架构，它集成了两级量化，以创建针对对抗性攻击的离散瓶颈，同时保留多模式推理能力。通过在像素补丁和语义层面离散化视觉表示，Q-MLLM阻止攻击途径并弥合跨模式安全对齐差距。我们的两阶段训练方法确保稳健的学习，同时保持模型效用。实验表明，与现有方法相比，Q-MLLM在针对越狱攻击和有毒图像攻击的防御成功率明显更高。值得注意的是，Q-MLLM在针对越狱攻击时实现了完美的防御成功率（100%），但在一种可发现的情况下，同时以最小的推理费用在多个实用工具基准上保持竞争性能。这项工作将载体量化建立为安全多模式人工智能系统的有效防御机制，而不需要昂贵的安全特定微调或检测费用。代码可在https://github.com/Amadeuszhao/QMLLM上获取。



## **3. PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization**

PSM：通过LLM引导的黑盒优化实现快速灵敏度最小化 cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16209v1) [paper-pdf](https://arxiv.org/pdf/2511.16209v1)

**Authors**: Huseein Jawad, Nicolas Brunel

**Abstract**: System prompts are critical for guiding the behavior of Large Language Models (LLMs), yet they often contain proprietary logic or sensitive information, making them a prime target for extraction attacks. Adversarial queries can successfully elicit these hidden instructions, posing significant security and privacy risks. Existing defense mechanisms frequently rely on heuristics, incur substantial computational overhead, or are inapplicable to models accessed via black-box APIs. This paper introduces a novel framework for hardening system prompts through shield appending, a lightweight approach that adds a protective textual layer to the original prompt. Our core contribution is the formalization of prompt hardening as a utility-constrained optimization problem. We leverage an LLM-as-optimizer to search the space of possible SHIELDs, seeking to minimize a leakage metric derived from a suite of adversarial attacks, while simultaneously preserving task utility above a specified threshold, measured by semantic fidelity to baseline outputs. This black-box, optimization-driven methodology is lightweight and practical, requiring only API access to the target and optimizer LLMs. We demonstrate empirically that our optimized SHIELDs significantly reduce prompt leakage against a comprehensive set of extraction attacks, outperforming established baseline defenses without compromising the model's intended functionality. Our work presents a paradigm for developing robust, utility-aware defenses in the escalating landscape of LLM security. The code is made public on the following link: https://github.com/psm-defense/psm

摘要: 系统提示对于指导大型语言模型（LLM）的行为至关重要，但它们通常包含专有逻辑或敏感信息，使其成为提取攻击的主要目标。对抗性查询可以成功地引出这些隐藏指令，从而构成重大的安全和隐私风险。现有的防御机制通常依赖于启发式方法，会产生大量的计算负担，或者不适用于通过黑匣子API访问的模型。本文介绍了一种通过屏蔽附加来强化系统提示的新颖框架，这是一种轻量级方法，可以在原始提示中添加保护性文本层。我们的核心贡献是将即时硬化形式化为一个受效用约束的优化问题。我们利用LLM作为优化器来搜索可能的SHIELD的空间，寻求最大限度地减少从一系列对抗攻击中获得的泄漏指标，同时将任务效用保持在指定阈值以上，该阈值通过基线输出的语义保真度来衡量。这种黑匣子、优化驱动的方法是轻量级且实用的，仅需要API访问目标和优化器LLM。我们通过经验证明，我们优化的SHIELD显着减少了针对一系列全面提取攻击的即时泄漏，在不损害模型预期功能的情况下优于既定的基线防御。我们的工作提供了一个在LLM安全不断升级的环境中开发强大的、实用程序感知的防御的范式。该代码在以下链接上公开：https://github.com/psm-defense/psm



## **4. When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models**

当对齐失败时：对视觉-语言-动作模型的多模式对抗攻击 cs.CV

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16203v1) [paper-pdf](https://arxiv.org/pdf/2511.16203v1)

**Authors**: Yuping Yan, Yuhan Xie, Yinxin Zhang, Lingjuan Lyu, Yaochu Jin

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.

摘要: 视觉-语言-动作模型（VLA）最近在具体环境中取得了显着进展，使机器人能够通过统一的多模式理解来感知、推理和行动。尽管它们的能力令人印象深刻，但这些系统的对抗鲁棒性在很大程度上仍未得到探索，尤其是在现实的多模式和黑匣子条件下。现有的研究主要关注单模式扰动，而忽视了从根本上影响体现推理和决策的跨模式失调。本文介绍了VLA-Fool，这是对白盒和黑盒设置下具体VLA模型中多模式对抗鲁棒性的全面研究。VLA-Fool统一了三个级别的多模式对抗攻击：（1）通过基于梯度和基于预算的操纵进行文本扰动，（2）通过补丁和噪音失真进行视觉扰动，以及（3）故意破坏感知和指令之间的语义对应性的跨模式失准攻击。我们进一步将VLA感知的语义空间融入到语言提示中，开发了第一个自动制作和语义引导的提示框架。使用微调的OpenVLA模型对LIBERO基准进行的实验表明，即使是微小的多峰扰动也会导致显着的行为偏差，这表明了体现多峰对齐的脆弱性。



## **5. An Image Is Worth Ten Thousand Words: Verbose-Text Induction Attacks on VLMs**

一张图片胜过一万个单词：对VLMs的动词文本归纳攻击 cs.CV

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16163v1) [paper-pdf](https://arxiv.org/pdf/2511.16163v1)

**Authors**: Zhi Luo, Zenghui Yuan, Wenqi Wei, Daizong Liu, Pan Zhou

**Abstract**: With the remarkable success of Vision-Language Models (VLMs) on multimodal tasks, concerns regarding their deployment efficiency have become increasingly prominent. In particular, the number of tokens consumed during the generation process has emerged as a key evaluation metric.Prior studies have shown that specific inputs can induce VLMs to generate lengthy outputs with low information density, which significantly increases energy consumption, latency, and token costs. However, existing methods simply delay the occurrence of the EOS token to implicitly prolong output, and fail to directly maximize the output token length as an explicit optimization objective, lacking stability and controllability.To address these limitations, this paper proposes a novel verbose-text induction attack (VTIA) to inject imperceptible adversarial perturbations into benign images via a two-stage framework, which identifies the most malicious prompt embeddings for optimizing and maximizing the output token of the perturbed images.Specifically, we first perform adversarial prompt search, employing reinforcement learning strategies to automatically identify adversarial prompts capable of inducing the LLM component within VLMs to produce verbose outputs. We then conduct vision-aligned perturbation optimization to craft adversarial examples on input images, maximizing the similarity between the perturbed image's visual embeddings and those of the adversarial prompt, thereby constructing malicious images that trigger verbose text generation. Comprehensive experiments on four popular VLMs demonstrate that our method achieves significant advantages in terms of effectiveness, efficiency, and generalization capability.

摘要: 随着视觉语言模型（VLM）在多模式任务上取得的巨大成功，对其部署效率的担忧变得越来越突出。特别是，生成过程中消耗的代币数量已成为一个关键的评估指标。之前的研究表明，特定的输入可能会导致VLM生成信息密度较低的冗长输出，从而显着增加能源消耗、延迟和代币成本。然而，现有方法只是延迟EOS令牌的出现以隐式延长输出，并且未能直接将输出令牌长度最大化作为显式优化目标，缺乏稳定性和可控性。为了解决这些局限性，本文提出了一种新型的动词文本诱导攻击（VTIA），通过两阶段框架将难以察觉的对抗性扰动注入良性图像中，它识别最恶意的提示嵌入，以优化和最大化受干扰图像的输出令牌。具体来说，我们首先执行对抗性提示搜索，采用强化学习策略自动识别能够在VLM中诱导LLM组件以产生详细输出的对抗性提示。然后，我们进行视觉对齐的扰动优化，在输入图像上制作对抗性示例，最大化受扰动图像的视觉嵌入与对抗性提示的视觉嵌入之间的相似性，从而构建触发冗长文本生成的恶意图像。对四种流行的VLM的综合实验表明，我们的方法在有效性、效率和概括能力方面取得了显着优势。



## **6. Layer-wise Noise Guided Selective Wavelet Reconstruction for Robust Medical Image Segmentation**

逐层噪音引导的选择性子波重建用于鲁棒医学图像分割 cs.CV

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16162v1) [paper-pdf](https://arxiv.org/pdf/2511.16162v1)

**Authors**: Yuting Lu, Ziliang Wang, Weixin Xu, Wei Zhang, Yongqiang Zhao, Yang Yu, Xiaohong Zhang

**Abstract**: Clinical deployment requires segmentation models to stay stable under distribution shifts and perturbations. The mainstream solution is adversarial training (AT) to improve robustness; however, AT often brings a clean--robustness trade-off and high training/tuning cost, which limits scalability and maintainability in medical imaging. We propose \emph{Layer-wise Noise-Guided Selective Wavelet Reconstruction (LNG-SWR)}. During training, we inject small, zero-mean noise at multiple layers to learn a frequency-bias prior that steers representations away from noise-sensitive directions. We then apply prior-guided selective wavelet reconstruction on the input/feature branch to achieve frequency adaptation: suppress noise-sensitive bands, enhance directional structures and shape cues, and stabilize boundary responses while maintaining spectral consistency. The framework is backbone-agnostic and adds low additional inference overhead. It can serve as a plug-in enhancement to AT and also improves robustness without AT. On CT and ultrasound datasets, under a unified protocol with PGD-$L_{\infty}/L_{2}$ and SSAH, LNG-SWR delivers consistent gains on clean Dice/IoU and significantly reduces the performance drop under strong attacks; combining LNG-SWR with AT yields additive gains. When combined with adversarial training, robustness improves further without sacrificing clean accuracy, indicating an engineering-friendly and scalable path to robust segmentation. These results indicate that LNG-SWR provides a simple, effective, and engineering-friendly path to robust medical image segmentation in both adversarial and standard training regimes.

摘要: 临床部署要求分割模型在分布变化和扰动下保持稳定。主流的解决方案是对抗训练（AT）来提高鲁棒性;然而，AT通常会带来一个干净的鲁棒性权衡和高训练/调整成本，这限制了医学成像的可扩展性和可维护性。本文提出了一种分层噪声引导的选择性小波重构算法（LNG-SWR）。在训练过程中，我们在多层注入小的零均值噪声，以学习频率偏置先验，使表示远离噪声敏感方向。然后，我们在输入/特征分支上应用优先引导的选择性子波重建，以实现频率自适应：抑制噪音敏感频段、增强方向结构和形状线索，并稳定边界响应，同时保持频谱一致性。该框架是主干不可知的，并且增加了较低的额外推断负担。它可以作为AT的插件增强，并且在没有AT的情况下也可以提高稳健性。在CT和超声数据集上，在PVD-$L_{\infty}/L_{2}$和SSAH的统一协议下，LNG-SWR在干净的Dice/IoU上提供了一致的收益，并显着降低了强攻击下的性能下降; LNG-SWR与AT相结合可以产生附加收益。当与对抗训练相结合时，鲁棒性会进一步提高，而不会牺牲清晰的准确性，这表明了一条工程友好且可扩展的鲁棒分割路径。这些结果表明，LNG-SWR为对抗和标准训练方案中的稳健医学图像分割提供了一种简单、有效且工程友好的路径。



## **7. SceneGuard: Training-Time Voice Protection with Scene-Consistent Audible Background Noise**

SceneGuard：训练时语音保护，具有场景一致的可听背景噪音 cs.SD

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16114v1) [paper-pdf](https://arxiv.org/pdf/2511.16114v1)

**Authors**: Rui Sang, Yuxuan Liu

**Abstract**: Voice cloning technology poses significant privacy threats by enabling unauthorized speech synthesis from limited audio samples. Existing defenses based on imperceptible adversarial perturbations are vulnerable to common audio preprocessing such as denoising and compression. We propose SceneGuard, a training-time voice protection method that applies scene-consistent audible background noise to speech recordings. Unlike imperceptible perturbations, SceneGuard leverages naturally occurring acoustic scenes (e.g., airport, street, park) to create protective noise that is contextually appropriate and robust to countermeasures. We evaluate SceneGuard on text-to-speech training attacks, demonstrating 5.5% speaker similarity degradation with extremely high statistical significance (p < 10^{-15}, Cohen's d = 2.18) while preserving 98.6% speech intelligibility (STOI = 0.986). Robustness evaluation shows that SceneGuard maintains or enhances protection under five common countermeasures including MP3 compression, spectral subtraction, lowpass filtering, and downsampling. Our results suggest that audible, scene-consistent noise provides a more robust alternative to imperceptible perturbations for training-time voice protection. The source code are available at: https://github.com/richael-sang/SceneGuard.

摘要: 语音克隆技术通过从有限的音频样本进行未经授权的语音合成，构成了严重的隐私威胁。基于不可感知的对抗扰动的现有防御很容易受到去噪和压缩等常见音频预处理的影响。我们提出SceneGuard，这是一种训练时的语音保护方法，可将场景一致的可听背景噪音应用于语音记录。与不可感知的扰动不同，SceneGuard利用自然发生的声学场景（例如，机场、街道、公园）以产生适合环境且对应对措施稳健的保护性噪音。我们评估了SceneGuard的文本转语音训练攻击，证明说话者相似性下降了5.5%，具有极高的统计学意义（p < 10 '' s d = 2.18），同时保留了98.6%的语音清晰度（STIOI = 0.986）。稳健性评估表明，SceneGuard在五种常见对策下保持或增强了保护，包括MP3压缩、频谱减法、低通过滤和下采样。我们的结果表明，可听的、场景一致的噪音为训练时的语音保护提供了一种更强大的可感知干扰的替代方案。源代码可访问：https://github.com/richael-sang/SceneGuard。



## **8. Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models**

多面攻击：暴露配备防御的视觉语言模型中的跨模型漏洞 cs.CR

AAAI 2026 Oral

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16110v1) [paper-pdf](https://arxiv.org/pdf/2511.16110v1)

**Authors**: Yijun Yang, Lichao Wang, Jianping Zhang, Chi Harold Liu, Lanqing Hong, Qiang Xu

**Abstract**: The growing misuse of Vision-Language Models (VLMs) has led providers to deploy multiple safeguards, including alignment tuning, system prompts, and content moderation. However, the real-world robustness of these defenses against adversarial attacks remains underexplored. We introduce Multi-Faceted Attack (MFA), a framework that systematically exposes general safety vulnerabilities in leading defense-equipped VLMs such as GPT-4o, Gemini-Pro, and Llama-4. The core component of MFA is the Attention-Transfer Attack (ATA), which hides harmful instructions inside a meta task with competing objectives. We provide a theoretical perspective based on reward hacking to explain why this attack succeeds. To improve cross-model transferability, we further introduce a lightweight transfer-enhancement algorithm combined with a simple repetition strategy that jointly bypasses both input-level and output-level filters without model-specific fine-tuning. Empirically, we show that adversarial images optimized for one vision encoder transfer broadly to unseen VLMs, indicating that shared visual representations create a cross-model safety vulnerability. Overall, MFA achieves a 58.5% success rate and consistently outperforms existing methods. On state-of-the-art commercial models, MFA reaches a 52.8% success rate, surpassing the second-best attack by 34%. These results challenge the perceived robustness of current defense mechanisms and highlight persistent safety weaknesses in modern VLMs. Code: https://github.com/cure-lab/MultiFacetedAttack

摘要: 视觉语言模型（VLM）的滥用日益严重，导致提供商部署多种保护措施，包括对齐调整、系统提示和内容审核。然而，这些防御系统针对对抗性攻击的真实鲁棒性仍然没有得到充分的研究。我们引入了多面攻击（MFA），这是一个框架，可以系统性地暴露领先的防御装备VLM（例如GPT-4 o、Gemini-Pro和Llama-4）中的一般安全漏洞。MFA的核心组件是注意力转移攻击（ATA），它将有害指令隐藏在具有相互竞争目标的Meta任务中。我们提供了一个基于奖励黑客的理论视角来解释这次攻击为何会成功。为了提高跨模型的可移植性，我们进一步引入了一种轻量级的传输增强算法，并结合简单的重复策略，该策略可以共同绕过输入级和输出级过滤器，而无需进行特定于模型的微调。从经验上看，我们表明，针对一个视觉编码器优化的对抗图像会广泛地转移到不可见的VLM，这表明共享的视觉表示会创建跨模型安全漏洞。总体而言，MFA的成功率达到了58.5%，并且始终优于现有方法。在最先进的商业模型上，MFA的成功率达到了52.8%，超过了第二好的攻击34%。这些结果挑战了当前防御机制的稳健性，并凸显了现代VLM中持续存在的安全弱点。代码：https://github.com/cure-lab/MultiFacetedAttack



## **9. Future-Back Threat Modeling: A Foresight-Driven Security Framework**

未来威胁建模：前瞻性驱动的安全框架 cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16088v1) [paper-pdf](https://arxiv.org/pdf/2511.16088v1)

**Authors**: Vu Van Than

**Abstract**: Traditional threat modeling remains reactive-focused on known TTPs and past incident data, while threat prediction and forecasting frameworks are often disconnected from operational or architectural artifacts. This creates a fundamental weakness: the most serious cyber threats often do not arise from what is known, but from what is assumed, overlooked, or not yet conceived, and frequently originate from the future, such as artificial intelligence, information warfare, and supply chain attacks, where adversaries continuously develop new exploits that can bypass defenses built on current knowledge. To address this mental gap, this paper introduces the theory and methodology of Future-Back Threat Modeling (FBTM). This predictive approach begins with envisioned future threat states and works backward to identify assumptions, gaps, blind spots, and vulnerabilities in the current defense architecture, providing a clearer and more accurate view of impending threats so that we can anticipate their emergence and shape the future we want through actions taken now. The proposed methodology further aims to reveal known unknowns and unknown unknowns, including tactics, techniques, and procedures that are emerging, anticipated, and plausible. This enhances the predictability of adversary behavior, particularly under future uncertainty, helping security leaders make informed decisions today that shape more resilient security postures for the future.

摘要: 传统的威胁建模仍然以已知的TTP和过去的事件数据为中心，而威胁预测和预测框架通常与操作或架构工件脱节。这造成了一个根本性的弱点：最严重的网络威胁往往不是来自已知的，而是来自假设、忽视或尚未构想的，并且通常起源于未来，例如人工智能、信息战和供应链攻击，对手不断开发新的漏洞，可以绕过基于当前知识的防御。为了解决这一心理差距，本文介绍了未来反向威胁建模（FBTM）的理论和方法论。这种预测方法从设想的未来威胁状态开始，并向后工作以识别当前防御架构中的假设、差距、盲点和漏洞，为即将发生的威胁提供更清晰、更准确的视图，以便我们能够预测它们的出现并塑造我们想要的未来通过现在采取的行动。拟议的方法论进一步旨在揭示已知的未知数和未知的未知数，包括正在出现的、预期的和合理的策略、技术和程序。这增强了对手行为的可预测性，特别是在未来不确定性的情况下，帮助安全领导者今天做出明智的决定，为未来塑造更具弹性的安全姿态。



## **10. Physically Realistic Sequence-Level Adversarial Clothing for Robust Human-Detection Evasion**

物理真实的序列级对抗服装，用于鲁棒的人体检测逃避 cs.CV

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16020v1) [paper-pdf](https://arxiv.org/pdf/2511.16020v1)

**Authors**: Dingkun Zhou, Patrick P. K. Chan, Hengxu Wu, Shikang Zheng, Ruiqi Huang, Yuanjie Zhao

**Abstract**: Deep neural networks used for human detection are highly vulnerable to adversarial manipulation, creating safety and privacy risks in real surveillance environments. Wearable attacks offer a realistic threat model, yet existing approaches usually optimize textures frame by frame and therefore fail to maintain concealment across long video sequences with motion, pose changes, and garment deformation. In this work, a sequence-level optimization framework is introduced to generate natural, printable adversarial textures for shirts, trousers, and hats that remain effective throughout entire walking videos in both digital and physical settings. Product images are first mapped to UV space and converted into a compact palette and control-point parameterization, with ICC locking to keep all colors printable. A physically based human-garment pipeline is then employed to simulate motion, multi-angle camera viewpoints, cloth dynamics, and illumination variation. An expectation-over-transformation objective with temporal weighting is used to optimize the control points so that detection confidence is minimized across whole sequences. Extensive experiments demonstrate strong and stable concealment, high robustness to viewpoint changes, and superior cross-model transferability. Physical garments produced with sublimation printing achieve reliable suppression under indoor and outdoor recordings, confirming real-world feasibility.

摘要: 用于人类检测的深度神经网络极易受到对抗性操纵的影响，从而在真实监控环境中造成安全和隐私风险。可穿戴攻击提供了一个现实的威胁模型，但现有的方法通常逐帧优化纹理，因此无法在具有运动、姿势变化和服装变形的长视频序列中保持隐藏。在这项工作中，引入了序列级优化框架，为衬衫、裤子和帽子生成自然、可打印的对抗纹理，这些纹理在数字和物理环境中的整个行走视频中仍然有效。产品图像首先映射到紫外空间，然后转换为紧凑的调色板和控制点参数化，并通过ICC锁定以保持所有颜色可打印。然后使用基于物理的人体-服装管道来模拟运动、多角度摄像机视角、布料动态和照明变化。使用具有时间加权的期望转换目标来优化控制点，以便在整个序列中最小化检测置信度。大量实验证明了强大且稳定的隐藏性、对观点变化的高鲁棒性以及卓越的跨模型可移植性。采用升华印花生产的物理服装在室内和室外记录下实现了可靠的抑制，证实了现实世界的可行性。



## **11. Nonadaptive One-Way to Hiding Implies Adaptive Quantum Reprogramming**

非自适应单向隐藏意味着自适应量子重编程 quant-ph

24 pages, 12 figures

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16009v1) [paper-pdf](https://arxiv.org/pdf/2511.16009v1)

**Authors**: Joseph Jaeger

**Abstract**: An important proof technique in the random oracle model involves reprogramming it on hard to predict inputs and arguing that an attacker cannot detect that this occurred. In the quantum setting, a particularly challenging version of this considers adaptive reprogramming wherein the points to be reprogrammed (or the output values they should be programmed to) are dependent on choices made by the adversary. Some quantum frameworks for analyzing adaptive reprogramming were given by Unruh (CRYPTO 2014, EUROCRYPT 2015), Grilo-Hövelmanns-Hülsing-Majenz (ASIACRYPT 2021), and Pan-Zeng (PKC 2024). We show, counterintuitively, that these adaptive results follow from the \emph{nonadaptive} one-way to hiding theorem of Ambainis-Hamburg-Unruh (CRYPTO 2019). These implications contradict beliefs (whether stated explicitly or implicitly) that some properties of the adaptive frameworks cannot be provided by the Ambainis-Hamburg-Unruh result.

摘要: 随机预言模型中的一项重要证明技术涉及对难以预测的输入进行重新编程，并认为攻击者无法检测到这种情况的发生。在量子环境中，一个特别具有挑战性的版本考虑自适应重新编程，其中要重新编程的点（或它们应该被编程到的输出值）取决于对手做出的选择。Unruh（CLARPTO 2014，EUROSYS PT 2015）、Grilo-Hövelmanns-Hülsing-Majenz（ASIACRYPT 2021）和Pan-Zeng（PKC 2024）给出了一些用于分析自适应重编程的量子框架。我们表明，与直觉相反，这些自适应结果源自Ambainis-Hamburg-Unruh的单向隐藏定理（CLARPTO 2019）。这些含义与Ambainis-Hamburg-Unruh结果无法提供自适应框架的某些属性的信念（无论是明确还是隐含地陈述）相矛盾。



## **12. Lifefin: Escaping Mempool Explosions in DAG-based BFT**

Lifefin：逃离基于DAB的BFT中的Mempool爆炸 cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15936v1) [paper-pdf](https://arxiv.org/pdf/2511.15936v1)

**Authors**: Jianting Zhang, Sen Yang, Alberto Sonnino, Sebastián Loza, Aniket Kate

**Abstract**: Directed Acyclic Graph (DAG)-based Byzantine Fault-Tolerant (BFT) protocols have emerged as promising solutions for high-throughput blockchains. By decoupling data dissemination from transaction ordering and constructing a well-connected DAG in the mempool, these protocols enable zero-message ordering and implicit view changes. However, we identify a fundamental liveness vulnerability: an adversary can trigger mempool explosions to prevent transaction commitment, ultimately compromising the protocol's liveness.   In response, this work presents Lifefin, a generic and self-stabilizing protocol designed to integrate seamlessly with existing DAG-based BFT protocols and circumvent such vulnerabilities. Lifefin leverages the Agreement on Common Subset (ACS) mechanism, allowing nodes to escape mempool explosions by committing transactions with bounded resource usage even in adverse conditions. As a result, Lifefin imposes (almost) zero overhead in typical cases while effectively eliminating liveness vulnerabilities.   To demonstrate the effectiveness of Lifefin, we integrate it into two state-of-the-art DAG-based BFT protocols, Sailfish and Mysticeti, resulting in two enhanced variants: Sailfish-Lifefin and Mysticeti-Lifefin. We implement these variants and compare them with the original Sailfish and Mysticeti systems. Our evaluation demonstrates that Lifefin achieves comparable transaction throughput while introducing only minimal additional latency to resist similar attacks.

摘要: 基于有向无环图（DAB）的拜占庭故障容忍（BFT）协议已成为高吞吐量区块链的有前途的解决方案。通过将数据传播与事务排序脱钩并在内存池中构建连接良好的DAB，这些协议实现了零消息排序和隐式视图更改。然而，我们发现了一个根本的活跃性漏洞：对手可以触发成员池爆炸以阻止事务承诺，最终损害协议的活跃性。   作为回应，这项工作提出了Lifefin，这是一种通用的自稳定协议，旨在与现有的基于DAB的BFT协议无缝集成并规避此类漏洞。Lifefin利用公共子集协议（ACS）机制，允许节点即使在不利条件下也通过提交具有有限资源使用量的事务来避免成员池爆炸。因此，Lifefin在典型情况下（几乎）实行零管理，同时有效地消除了活力漏洞。   为了证明Lifefin的有效性，我们将其集成到两个最先进的基于DAB的BFT协议Sailfish和Mysticeti中，从而产生了两个增强的变体：Sailfish-Lifefin和Mysticeti-Lifefin。我们实现这些变体并将它们与原始的Sailfish和Mysticeti系统进行比较。我们的评估表明，Lifefin实现了相当的交易吞吐量，同时仅引入了最小的额外延迟来抵抗类似的攻击。



## **13. Cyber-Resilient Data-Driven Event-Triggered Secure Control for Autonomous Vehicles Under False Data Injection Attacks**

虚假数据注入攻击下自动驾驶汽车的网络弹性数据驱动事件触发安全控制 eess.SY

14 pages, 8 figures

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15925v1) [paper-pdf](https://arxiv.org/pdf/2511.15925v1)

**Authors**: Yashar Mousavi, Mahsa Tavasoli, Ibrahim Beklan Kucukdemiral, Umit Cali, Abdolhossein Sarrafzadeh, Ali Karimoddini, Afef Fekih

**Abstract**: This paper proposes a cyber-resilient secure control framework for autonomous vehicles (AVs) subject to false data injection (FDI) threats as actuator attacks. The framework integrates data-driven modeling, event-triggered communication, and fractional-order sliding mode control (FSMC) to enhance the resilience against adversarial interventions. A dynamic model decomposition (DMD)-based methodology is employed to extract the lateral dynamics from real-world data, eliminating the reliance on conventional mechanistic modeling. To optimize communication efficiency, an event-triggered transmission scheme is designed to reduce the redundant transmissions while ensuring system stability. Furthermore, an extended state observer (ESO) is developed for real-time estimation and mitigation of actuator attack effects. Theoretical stability analysis, conducted using Lyapunov methods and linear matrix inequality (LMI) formulations, guarantees exponential error convergence. Extensive simulations validate the proposed event-triggered secure control framework, demonstrating substantial improvements in attack mitigation, communication efficiency, and lateral tracking performance. The results show that the framework effectively counteracts actuator attacks while optimizing communication-resource utilization, making it highly suitable for safety-critical AV applications.

摘要: 本文针对自动驾驶汽车（AV）提出了一种具有网络弹性的安全控制框架，该框架受到致动器攻击的虚假数据注入（FDI）威胁。该框架集成了数据驱动建模、事件触发通信和分数阶滑动模式控制（FSMC），以增强对抗性干预的弹性。采用基于动态模型分解（DMZ）的方法从现实世界数据中提取横向动力学，消除了对传统机械建模的依赖。为了优化通信效率，设计了事件触发传输方案，以减少冗余传输，同时确保系统稳定性。此外，还开发了扩展状态观测器（ESO）来实时估计和缓解致动器攻击效应。使用李雅普诺夫方法和线性矩阵不等式（LDI）公式进行的理论稳定性分析保证了指数误差收敛。广泛的模拟验证了拟议的事件触发安全控制框架，展示了攻击缓解、通信效率和横向跟踪性能方面的重大改进。结果表明，该框架有效地对抗致动器攻击，同时优化通信资源利用率，使其非常适合安全关键的AV应用。



## **14. TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models**

TopoReformer：在OCR模型中使用Topological purpose来缓解对抗攻击 cs.LG

Accepted at AAAI 2026 AI for CyberSecurity (AICS) Workshop

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15807v1) [paper-pdf](https://arxiv.org/pdf/2511.15807v1)

**Authors**: Bhagyesh Kumar, A S Aravinthakashan, Akshat Satyanarayan, Ishaan Gakhar, Ujjwal Verma

**Abstract**: Adversarially perturbed images of text can cause sophisticated OCR systems to produce misleading or incorrect transcriptions from seemingly invisible changes to humans. Some of these perturbations even survive physical capture, posing security risks to high-stakes applications such as document processing, license plate recognition, and automated compliance systems. Existing defenses, such as adversarial training, input preprocessing, or post-recognition correction, are often model-specific, computationally expensive, and affect performance on unperturbed inputs while remaining vulnerable to unseen or adaptive attacks. To address these challenges, TopoReformer is introduced, a model-agnostic reformation pipeline that mitigates adversarial perturbations while preserving the structural integrity of text images. Topology studies properties of shapes and spaces that remain unchanged under continuous deformations, focusing on global structures such as connectivity, holes, and loops rather than exact distance. Leveraging these topological features, TopoReformer employs a topological autoencoder to enforce manifold-level consistency in latent space and improve robustness without explicit gradient regularization. The proposed method is benchmarked on EMNIST, MNIST, against standard adversarial attacks (FGSM, PGD, Carlini-Wagner), adaptive attacks (EOT, BDPA), and an OCR-specific watermark attack (FAWA).

摘要: 文本图像受到不利干扰可能会导致复杂的OCR系统从人类看似不可见的变化中产生误导性或不正确的转录。其中一些扰动甚至可以在物理捕获中幸存下来，从而对文档处理、车牌识别和自动合规系统等高风险应用构成安全风险。现有的防御措施，例如对抗训练、输入预处理或识别后纠正，通常是特定于模型的，计算昂贵的，并且会影响未受干扰输入的性能，同时仍然容易受到不可见或适应性攻击。为了应对这些挑战，引入了TopoReformer，这是一个模型不可知的重组管道，可以减轻对抗性扰动，同时保留文本图像的结构完整性。Topology研究在连续变形下保持不变的形状和空间的属性，重点关注全球结构，例如连通性、洞和环，而不是精确的距离。利用这些拓扑特征，TopoReformer采用拓扑自动编码器来强制潜在空间中的总管级一致性，并在无需显式梯度正规化的情况下提高鲁棒性。所提出的方法以EMBIST、MNIST为基准，针对标准对抗攻击（FGSM、PVD、Carlini-Wagner）、自适应攻击（OT、BCPA）和OCR特定水印攻击（FAWA）。



## **15. Transferable Dual-Domain Feature Importance Attack against AI-Generated Image Detector**

针对人工智能生成图像检测器的可转移双域特征重要性攻击 cs.CV

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15571v1) [paper-pdf](https://arxiv.org/pdf/2511.15571v1)

**Authors**: Weiheng Zhu, Gang Cao, Jing Liu, Lifang Yu, Shaowei Weng

**Abstract**: Recent AI-generated image (AIGI) detectors achieve impressive accuracy under clean condition. In view of antiforensics, it is significant to develop advanced adversarial attacks for evaluating the security of such detectors, which remains unexplored sufficiently. This letter proposes a Dual-domain Feature Importance Attack (DuFIA) scheme to invalidate AIGI detectors to some extent. Forensically important features are captured by the spatially interpolated gradient and frequency-aware perturbation. The adversarial transferability is enhanced by jointly modeling spatial and frequency-domain feature importances, which are fused to guide the optimization-based adversarial example generation. Extensive experiments across various AIGI detectors verify the cross-model transferability, transparency and robustness of DuFIA.

摘要: 最近的人工智能生成图像（AIGI）检测器在干净的条件下实现了令人印象深刻的准确性。鉴于反取证学，开发高级对抗攻击来评估此类检测器的安全性非常重要，而这一点仍然没有得到充分的探索。这封信提出了一种双域特征重要性攻击（DuFIA）方案，以在某种程度上使AIGI检测器无效。通过空间内插梯度和频率感知扰动来捕获具有法医意义的特征。通过联合建模空间和频域特征重要性来增强对抗性的可移植性，并将其融合以指导基于优化的对抗性示例生成。各种AIGI检测器的广泛实验验证了DuFIA的跨模型可移植性、透明度和稳健性。



## **16. Beluga: Block Synchronization for BFT Consensus Protocols**

Beluga：BFT共识协议的块同步 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15517v1) [paper-pdf](https://arxiv.org/pdf/2511.15517v1)

**Authors**: Tasos Kichidis, Lefteris Kokoris-Kogias, Arun Koshy, Ilya Sergey, Alberto Sonnino, Mingwei Tian, Jianting Zhang

**Abstract**: Modern high-throughput BFT consensus protocols use streamlined push-pull mechanisms to disseminate blocks and keep happy-path performance optimal. Yet state-of-the-art designs lack a principled and efficient way to exchange blocks, which leaves them open to targeted attacks and performance collapse under network asynchrony. This work introduces the concept of a block synchronizer, a simple abstraction that drives incremental block retrieval and enforces resource-aware exchange. Its interface and role fit cleanly inside a modern BFT consensus stack. We also uncover a new attack, where an adversary steers honest validators into redundant, uncoordinated pulls that exhaust bandwidth and stall progress. Beluga is a modular and scarcity-aware instantiation of the block synchronizer. It achieves optimal common-case latency while bounding the cost of recovery under faults and adversarial behavior. We integrate Beluga into Mysticeti, the consensus core of the Sui blockchain, and show on a geo-distributed AWS deployment that Beluga sustains optimal performance in the optimistic path and, under attack, delivers up to 3x higher throughput and 25x lower latency than prior designs. The Sui blockchain adopted Beluga in production.

摘要: 现代高吞吐量BFT共识协议使用简化的推拉机制来传播块并保持幸福路径性能最佳。然而，最先进的设计缺乏原则性且有效的交换块的方法，这使得它们容易受到有针对性的攻击，并在网络混乱下性能崩溃。这项工作引入了块同步器的概念，这是一个简单的抽象，可以驱动增量块检索并执行资源感知交换。它的界面和角色完全适合现代BFT共识堆栈。我们还发现了一种新的攻击，对手引导诚实的验证者进入冗余、不协调的拉入，从而耗尽带宽并阻碍进展。Beluga是块同步器的模块化且具有稀缺性的实例。它实现了最佳的常见情况延迟，同时限制了故障和对抗行为下的恢复成本。我们将Beluga集成到Sui区块链的共识核心Mysticeti中，并在地理分布的AWS部署中显示，Beluga在乐观路径中保持最佳性能，并且在攻击下，提供比以前设计高3倍的吞吐量和低25倍的延迟。Sui区块链在生产中采用了白鲸。



## **17. HV-Attack: Hierarchical Visual Attack for Multimodal Retrieval Augmented Generation**

HV-Attack：多模态检索增强生成的层次视觉攻击 cs.CV

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15435v1) [paper-pdf](https://arxiv.org/pdf/2511.15435v1)

**Authors**: Linyin Luo, Yujuan Ding, Yunshan Ma, Wenqi Fan, Hanjiang Lai

**Abstract**: Advanced multimodal Retrieval-Augmented Generation (MRAG) techniques have been widely applied to enhance the capabilities of Large Multimodal Models (LMMs), but they also bring along novel safety issues. Existing adversarial research has revealed the vulnerability of MRAG systems to knowledge poisoning attacks, which fool the retriever into recalling injected poisoned contents. However, our work considers a different setting: visual attack of MRAG by solely adding imperceptible perturbations at the image inputs of users, without manipulating any other components. This is challenging due to the robustness of fine-tuned retrievers and large-scale generators, and the effect of visual perturbation may be further weakened by propagation through the RAG chain. We propose a novel Hierarchical Visual Attack that misaligns and disrupts the two inputs (the multimodal query and the augmented knowledge) of MRAG's generator to confuse its generation. We further design a hierarchical two-stage strategy to obtain misaligned augmented knowledge. We disrupt the image input of the retriever to make it recall irrelevant knowledge from the original database, by optimizing the perturbation which first breaks the cross-modal alignment and then disrupts the multimodal semantic alignment. We conduct extensive experiments on two widely-used MRAG datasets: OK-VQA and InfoSeek. We use CLIP-based retrievers and two LMMs BLIP-2 and LLaVA as generators. Results demonstrate the effectiveness of our visual attack on MRAG through the significant decrease in both retrieval and generation performance.

摘要: 高级多模式检索增强生成（MRAG）技术已被广泛应用于增强大型多模式模型（LSYS）的能力，但它们也带来了新的安全问题。现有的对抗性研究揭示了MRAG系统容易受到知识中毒攻击，知识中毒攻击会欺骗寻回犬回忆注入的有毒内容。然而，我们的工作考虑了一种不同的设置：通过仅在用户的图像输入中添加难以感知的扰动来对MRAG进行视觉攻击，而无需操纵任何其他组件。由于微调检索器和大规模发生器的鲁棒性，这具有挑战性，并且视觉扰动的影响可能会通过RAG链的传播而进一步减弱。我们提出了一种新型的分层视觉攻击，它会错位和破坏MRAG生成器的两个输入（多模式查询和增强知识），以混淆其生成。我们进一步设计了分层的两阶段策略来获得错位的增强知识。我们通过优化扰动，首先打破跨模式对齐，然后破坏多模式语义对齐，扰乱检索器的图像输入，使其从原始数据库中回忆起不相关的知识。我们对两个广泛使用的MRAG数据集进行了广泛的实验：OK-VQA和InfoSeek。我们使用基于CLIP的回收器和两个LIP-BLIP-2和LLaVA作为发生器。结果表明，我们的视觉攻击的有效性MRAG通过检索和生成性能的显着下降。



## **18. Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models**

对抗性诗歌作为大型语言模型中通用的单轮越狱机制 cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15304v2) [paper-pdf](https://arxiv.org/pdf/2511.15304v2)

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.

摘要: 我们提供的证据表明，对抗性诗歌可以作为大型语言模型（LLM）的通用单轮越狱技术。在25个前沿专有和开放重量模型中，精心策划的诗意提示产生了很高的攻击成功率（ASB），一些提供商超过了90%。MLCommons和EU CoP风险分类的映射提示表明，诗意攻击跨CBRN、操纵、网络犯罪和失去控制领域转移。通过标准化元提示将1，200个MLCommons有害提示转换为诗句，产生的ASB比散文基线高出18倍。使用3名开放权重LLM评委的整体评估输出，他们的二元安全性评估在分层的人类标记子集上进行了验证。诗意框架的平均越狱成功率为62%，元提示转换的平均越狱成功率约为43%（与非诗意基线相比），大大优于非诗意基线，并揭示了示范家庭和安全培训方法之间的系统性弱点。这些研究结果表明，仅靠风格差异就可以规避当代安全机制，这表明当前对齐方法和评估协议存在根本性局限性。



## **19. Securing AI Agents Against Prompt Injection Attacks**

保护人工智能代理免受即时注入攻击 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15759v1) [paper-pdf](https://arxiv.org/pdf/2511.15759v1)

**Authors**: Badrinath Ramakrishnan, Akshaya Balaji

**Abstract**: Retrieval-augmented generation (RAG) systems have become widely used for enhancing large language model capabilities, but they introduce significant security vulnerabilities through prompt injection attacks. We present a comprehensive benchmark for evaluating prompt injection risks in RAG-enabled AI agents and propose a multi-layered defense framework. Our benchmark includes 847 adversarial test cases across five attack categories: direct injection, context manipulation, instruction override, data exfiltration, and cross-context contamination. We evaluate three defense mechanisms: content filtering with embedding-based anomaly detection, hierarchical system prompt guardrails, and multi-stage response verification, across seven state-of-the-art language models. Our combined framework reduces successful attack rates from 73.2% to 8.7% while maintaining 94.3% of baseline task performance. We release our benchmark dataset and defense implementation to support future research in AI agent security.

摘要: 检索增强生成（RAG）系统已被广泛用于增强大型语言模型能力，但它们通过提示注入攻击引入了严重的安全漏洞。我们提出了一个全面的基准来评估支持RAG的人工智能代理中的即时注入风险，并提出了一个多层防御框架。我们的基准测试包括跨越五种攻击类别的847个对抗测试案例：直接注入、上下文操纵、指令覆盖、数据溢出和跨上下文污染。我们评估了三种防御机制：基于嵌入的异常检测的内容过滤、分层系统提示护栏和跨七种最先进语言模型的多阶段响应验证。我们的组合框架将成功攻击率从73.2%降低到8.7%，同时保持94.3%的基线任务性能。我们发布了我们的基准数据集和防御实施，以支持未来的人工智能代理安全研究。



## **20. Adversarial Attack on Black-Box Multi-Agent by Adaptive Perturbation**

自适应扰动对黑匣子多智能体的对抗攻击 cs.MA

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15292v1) [paper-pdf](https://arxiv.org/pdf/2511.15292v1)

**Authors**: Jianming Chen, Yawen Wang, Junjie Wang, Xiaofei Xie, Yuanzhe Hu, Qing Wang, Fanjiang Xu

**Abstract**: Evaluating security and reliability for multi-agent systems (MAS) is urgent as they become increasingly prevalent in various applications. As an evaluation technique, existing adversarial attack frameworks face certain limitations, e.g., impracticality due to the requirement of white-box information or high control authority, and a lack of stealthiness or effectiveness as they often target all agents or specific fixed agents. To address these issues, we propose AdapAM, a novel framework for adversarial attacks on black-box MAS. AdapAM incorporates two key components: (1) Adaptive Selection Policy simultaneously selects the victim and determines the anticipated malicious action (the action would lead to the worst impact on MAS), balancing effectiveness and stealthiness. (2) Proxy-based Perturbation to Induce Malicious Action utilizes generative adversarial imitation learning to approximate the target MAS, allowing AdapAM to generate perturbed observations using white-box information and thus induce victims to execute malicious action in black-box settings. We evaluate AdapAM across eight multi-agent environments and compare it with four state-of-the-art and commonly-used baselines. Results demonstrate that AdapAM achieves the best attack performance in different perturbation rates. Besides, AdapAM-generated perturbations are the least noisy and hardest to detect, emphasizing the stealthiness.

摘要: 随着多代理系统（MAS）在各种应用中日益流行，评估其安全性和可靠性变得紧迫。作为一种评估技术，现有的对抗攻击框架面临着一定的局限性，例如，由于需要白盒信息或高控制权限，并且缺乏隐蔽性或有效性，因此不切实际，因为它们通常针对所有代理或特定固定代理。为了解决这些问题，我们提出了AdapAM，这是一种针对黑匣子MAS的对抗性攻击的新型框架。AdapAM包含两个关键组件：（1）自适应选择策略同时选择受害者并确定预期的恶意行为（该行为将对MAS造成最严重的影响），平衡有效性和隐蔽性。(2)基于代理的扰动诱导恶意动作利用生成式对抗模仿学习来逼近目标MAS，允许AdapAM使用白盒信息生成扰动观察，从而诱导受害者在黑匣子设置中执行恶意动作。我们在八个多代理环境中评估AdapAM，并将其与四个最先进且常用的基线进行比较。结果表明，AdapAM在不同的扰动率下都能达到最佳的攻击性能。此外，AdapAM生成的扰动噪音最小，也最难检测，强调了隐蔽性。



## **21. The Walls Have Ears: Unveiling Cross-Chain Sandwich Attacks in DeFi**

墙有耳朵：揭露DeFi中的跨链三明治攻击 cs.CE

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15245v1) [paper-pdf](https://arxiv.org/pdf/2511.15245v1)

**Authors**: Chuanlei Li, Zhicheng Sun, Jing Xin Yuu, Xuechao Wang

**Abstract**: Cross-chain interoperability is a core component of modern blockchain infrastructure, enabling seamless asset transfers and composable applications across multiple blockchain ecosystems. However, the transparency of cross-chain messages can inadvertently expose sensitive transaction information, creating opportunities for adversaries to exploit value through manipulation or front-running strategies.   In this work, we investigate cross-chain sandwich attacks targeting liquidity pool-based cross-chain bridge protocols. We uncover a critical vulnerability where attackers can exploit events emitted on the source chain to learn transaction details on the destination chain before they appear in the destination chain mempool. This information advantage allows attackers to strategically place front-running and back-running transactions, ensuring that their front-running transactions always precede those of existing MEV bots monitoring the mempool of the destination chain. Moreover, current sandwich-attack defenses are ineffective against this new cross-chain variant. To quantify this threat, we conduct an empirical study using two months (August 10 to October 10, 2025) of cross-chain transaction data from the Symbiosis protocol and a tailored heuristic detection model. Our analysis identifies attacks that collectively garnered over \(5.27\) million USD in profit, equivalent to 1.28\% of the total bridged volume.

摘要: 跨链互操作性是现代区块链基础设施的核心组件，可以实现跨多个区块链生态系统的无缝资产转移和可组合应用程序。然而，跨链消息的透明度可能会无意中暴露敏感的交易信息，为对手通过操纵或抢先策略利用价值创造机会。   在这项工作中，我们调查了针对基于流动性池的跨链桥协议的跨链三明治攻击。我们发现了一个关键漏洞，攻击者可以利用源链上发出的事件来了解目标链上的交易详细信息，然后才出现在目标链成员池中。这种信息优势使攻击者能够战略性地放置前置和后置事务，确保他们的前置事务始终领先于监控目标链成员池的现有MEV机器人的事务。此外，当前的网络攻击防御措施对这种新的跨链变体无效。为了量化这一威胁，我们使用来自共生协议的两个月（2025年8月10日至10月10日）的跨链交易数据和量身定制的启发式检测模型进行了一项实证研究。我们的分析确定了总共获得了超过527万美元利润的攻击，相当于总桥梁量的1.28%。



## **22. Trustworthy GenAI over 6G: Integrated Applications and Security Frameworks**

6G上值得信赖的GenAI：集成应用程序和安全框架 cs.CR

8 pages, 5 figures. Submitted for publication

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15206v1) [paper-pdf](https://arxiv.org/pdf/2511.15206v1)

**Authors**: Bui Duc Son, Trinh Van Chien, Dong In Kim

**Abstract**: The integration of generative artificial intelligence (GenAI) into 6G networks promises substantial performance gains while simultaneously exposing novel security vulnerabilities rooted in multimodal data processing and autonomous reasoning. This article presents a unified perspective on cross-domain vulnerabilities that arise across integrated sensing and communication (ISAC), federated learning (FL), digital twins (DTs), diffusion models (DMs), and large telecommunication models (LTMs). We highlight emerging adversarial agents such as compromised DTs and LTMs that can manipulate both the physical and cognitive layers of 6G systems. To address these risks, we propose an adaptive evolutionary defense (AED) concept that continuously co-evolves with attacks through GenAI-driven simulation and feedback, combining physical-layer protection, secure learning pipelines, and cognitive-layer resilience. A case study using an LLM-based port prediction model for fluid-antenna systems demonstrates the susceptibility of GenAI modules to adversarial perturbations and the effectiveness of the proposed defense concept. Finally, we summarize open challenges and future research directions toward building trustworthy, quantum-resilient, and adaptive GenAI-enabled 6G networks.

摘要: 将生成人工智能（GenAI）集成到6G网络中有望大幅提高性能，同时暴露出植根于多模式数据处理和自主推理的新型安全漏洞。本文从统一的角度探讨了集成传感和通信（ISAC）、联邦学习（FL）、数字双胞胎（DT）、扩散模型（DM）和大型电信模型（TLR）中出现的跨领域漏洞。我们重点介绍了新兴的对抗性代理，例如受损的DT和TLR，它们可以操纵6G系统的物理层和认知层。为了解决这些风险，我们提出了一种自适应进化防御（AED）概念，该概念通过GenAI驱动的模拟和反馈与攻击持续协同进化，结合了物理层保护、安全学习管道和认知层弹性。使用基于LLM的流体天线系统端口预测模型的案例研究展示了GenAI模块对对抗性扰动的敏感性以及拟议防御概念的有效性。最后，我们总结了构建值得信赖、具有量子弹性和自适应性的GenAI支持6G网络的开放挑战和未来研究方向。



## **23. Effective Code Membership Inference for Code Completion Models via Adversarial Prompts**

通过对抗性预算的代码完成模型的有效代码成员资格推断 cs.SE

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15107v1) [paper-pdf](https://arxiv.org/pdf/2511.15107v1)

**Authors**: Yuan Jiang, Zehao Li, Shan Huang, Christoph Treude, Xiaohong Su, Tiantian Wang

**Abstract**: Membership inference attacks (MIAs) on code completion models offer an effective way to assess privacy risks by inferring whether a given code snippet was part of the training data. Existing black- and gray-box MIAs rely on expensive surrogate models or manually crafted heuristic rules, which limit their ability to capture the nuanced memorization patterns exhibited by over-parameterized code language models. To address these challenges, we propose AdvPrompt-MIA, a method specifically designed for code completion models, combining code-specific adversarial perturbations with deep learning. The core novelty of our method lies in designing a series of adversarial prompts that induce variations in the victim code model's output. By comparing these outputs with the ground-truth completion, we construct feature vectors to train a classifier that automatically distinguishes member from non-member samples. This design allows our method to capture richer memorization patterns and accurately infer training set membership. We conduct comprehensive evaluations on widely adopted models, such as Code Llama 7B, over the APPS and HumanEval benchmarks. The results show that our approach consistently outperforms state-of-the-art baselines, with AUC gains of up to 102%. In addition, our method exhibits strong transferability across different models and datasets, underscoring its practical utility and generalizability.

摘要: 对代码完成模型的成员资格推理攻击（MIA）通过推断给定代码片段是否是训练数据的一部分，提供了一种评估隐私风险的有效方法。现有的黑盒和灰盒MIA依赖于昂贵的代理模型或手动构建的启发式规则，这限制了它们捕获过度参数化的代码语言模型所表现出的细致入微的记忆模式的能力。为了应对这些挑战，我们提出了Advobert-MIA，这是一种专门为代码完成模型设计的方法，将特定于代码的对抗性扰动与深度学习相结合。我们方法的核心新颖之处在于设计一系列对抗提示，这些提示会导致受害者代码模型的输出发生变化。通过将这些输出与地面真相完成进行比较，我们构建特征载体来训练自动区分成员样本与非成员样本的分类器。这种设计使我们的方法能够捕获更丰富的记忆模式并准确地推断训练集成员关系。我们根据APPS和HumanEval基准对广泛采用的模型（例如Code Llama 7 B）进行全面评估。结果表明，我们的方法始终优于最先进的基线，AUR收益高达102%。此外，我们的方法在不同模型和数据集之间表现出很强的可移植性，强调了其实际实用性和可推广性。



## **24. Critical Evaluation of Quantum Machine Learning for Adversarial Robustness**

量子机器学习对抗鲁棒性的批判性评估 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.14989v1) [paper-pdf](https://arxiv.org/pdf/2511.14989v1)

**Authors**: Saeefa Rubaiyet Nowmi, Jesus Lopez, Md Mahmudul Alam Imon, Shahrooz Pouryouse, Mohammad Saidur Rahman

**Abstract**: Quantum Machine Learning (QML) integrates quantum computational principles into learning algorithms, offering improved representational capacity and computational efficiency. Nevertheless, the security and robustness of QML systems remain underexplored, especially under adversarial conditions. In this paper, we present a systematization of adversarial robustness in QML, integrating conceptual organization with empirical evaluation across three threat models-black-box, gray-box, and white-box. We implement representative attacks in each category, including label-flipping for black-box, QUID encoder-level data poisoning for gray-box, and FGSM and PGD for white-box, using Quantum Neural Networks (QNNs) trained on two datasets from distinct domains: MNIST from computer vision and AZ-Class from Android malware, across multiple circuit depths (2, 5, 10, and 50 layers) and two encoding schemes (angle and amplitude). Our evaluation shows that amplitude encoding yields the highest clean accuracy (93% on MNIST and 67% on AZ-Class) in deep, noiseless circuits; however, it degrades sharply under adversarial perturbations and depolarization noise (p=0.01), dropping accuracy below 5%. In contrast, angle encoding, while offering lower representational capacity, remains more stable in shallow, noisy regimes, revealing a trade-off between capacity and robustness. Moreover, the QUID attack attains higher attack success rates, though quantum noise channels disrupt the Hilbert-space correlations it exploits, weakening its impact in image domains. This suggests that noise can act as a natural defense mechanism in Noisy Intermediate-Scale Quantum (NISQ) systems. Overall, our findings guide the development of secure and resilient QML architectures for practical deployment. These insights underscore the importance of designing threat-aware models that remain reliable under real-world noise in NISQ settings.

摘要: 量子机器学习（QML）将量子计算原理集成到学习算法中，提供改进的表示能力和计算效率。然而，QML系统的安全性和稳健性仍然没有得到充分的研究，尤其是在敌对条件下。在本文中，我们提出了QML中对抗鲁棒性的系统化，将概念组织与三种威胁模型（黑箱、灰箱和白箱）的经验评估集成在一起。我们使用在来自不同领域的两个数据集上训练的量子神经网络（QNN），在每个类别中实施代表性攻击，包括针对黑匣子的标签翻转、针对灰箱的QUID编码器级数据中毒以及针对白箱的FGSM和PGP：来自计算机视觉的MNIST和来自Android恶意软件的AZ-Class，跨越多个电路深度（2、5、10和50层）和两种编码方案（角度和幅度）。我们的评估表明，幅度编码在深度无噪电路中产生最高的清晰准确性（MNIST为93%，AZ-Class为67%）;然而，在对抗性扰动和去极化噪音下，它会急剧下降（p=0.01），准确性下降到5%以下。相比之下，角度编码虽然提供较低的代表容量，但在浅层、有噪音的区域中仍然更加稳定，揭示了容量和鲁棒性之间的权衡。此外，QUID攻击获得了更高的攻击成功率，尽管量子噪音通道扰乱了它所利用的Hilbert空间相关性，削弱了它在图像域中的影响。这表明噪音可以充当有噪的中规模量子（NISQ）系统中的自然防御机制。总体而言，我们的研究结果指导了安全且有弹性的QML架构的开发以进行实际部署。这些见解强调了设计威胁感知模型的重要性，这些模型在NISQ环境中在现实世界噪音下保持可靠。



## **25. Attacking Autonomous Driving Agents with Adversarial Machine Learning: A Holistic Evaluation with the CARLA Leaderboard**

利用对抗性机器学习攻击自动驾驶代理：CARLA排行榜的整体评估 cs.CR

12 pages

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14876v1) [paper-pdf](https://arxiv.org/pdf/2511.14876v1)

**Authors**: Henry Wong, Clement Fung, Weiran Lin, Karen Li, Stanley Chen, Lujo Bauer

**Abstract**: To autonomously control vehicles, driving agents use outputs from a combination of machine-learning (ML) models, controller logic, and custom modules. Although numerous prior works have shown that adversarial examples can mislead ML models used in autonomous driving contexts, it remains unclear if these attacks are effective at producing harmful driving actions for various agents, environments, and scenarios.   To assess the risk of adversarial examples to autonomous driving, we evaluate attacks against a variety of driving agents, rather than against ML models in isolation. To support this evaluation, we leverage CARLA, an urban driving simulator, to create and evaluate adversarial examples. We create adversarial patches designed to stop or steer driving agents, stream them into the CARLA simulator at runtime, and evaluate them against agents from the CARLA Leaderboard, a public repository of best-performing autonomous driving agents from an annual research competition. Unlike prior work, we evaluate attacks against autonomous driving systems without creating or modifying any driving-agent code and against all parts of the agent included with the ML model.   We perform a case-study investigation of two attack strategies against three open-source driving agents from the CARLA Leaderboard across multiple driving scenarios, lighting conditions, and locations. Interestingly, we show that, although some attacks can successfully mislead ML models into predicting erroneous stopping or steering commands, some driving agents use modules, such as PID control or GPS-based rules, that can overrule attacker-manipulated predictions from ML models.

摘要: 为了自主控制车辆，驾驶代理使用机器学习（ML）模型、控制器逻辑和自定义模块组合的输出。尽管许多先前的工作表明，对抗性示例可能会误导自动驾驶环境中使用的ML模型，但目前尚不清楚这些攻击是否有效地对各种代理、环境和场景产生有害驾驶行为。   为了评估自动驾驶对抗示例的风险，我们评估了针对各种驾驶代理的攻击，而不是孤立地针对ML模型的攻击。为了支持此评估，我们利用城市驾驶模拟器CARLA来创建和评估对抗性示例。我们创建对抗补丁，旨在阻止或引导驾驶代理，在运行时将它们流传输到CARLA模拟器中，并根据CARLA排行榜上的代理对其进行评估，CARLA排行榜是年度研究竞赛中表现最佳的自动驾驶代理的公共存储库。与之前的工作不同，我们评估针对自动驾驶系统的攻击，而无需创建或修改任何驾驶代理代码，以及针对ML模型中包含的代理的所有部分。   我们针对CARLA排行榜上的三种开源驾驶代理在多种驾驶场景、照明条件和地点进行了两种攻击策略的案例研究调查。有趣的是，我们表明，尽管一些攻击可以成功地误导ML模型预测错误的停止或转向命令，但一些驾驶代理使用模块（例如ID控制或基于GPS的规则），可以推翻ML模型中攻击者操纵的预测。



## **26. FLARE: Adaptive Multi-Dimensional Reputation for Robust Client Reliability in Federated Learning**

DART：自适应多维声誉，在联邦学习中实现稳健的客户端可靠性 cs.LG

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.14715v2) [paper-pdf](https://arxiv.org/pdf/2511.14715v2)

**Authors**: Abolfazl Younesi, Leon Kiss, Zahra Najafabadi Samani, Juan Aznar Poveda, Thomas Fahringer

**Abstract**: Federated learning (FL) enables collaborative model training while preserving data privacy. However, it remains vulnerable to malicious clients who compromise model integrity through Byzantine attacks, data poisoning, or adaptive adversarial behaviors. Existing defense mechanisms rely on static thresholds and binary classification, failing to adapt to evolving client behaviors in real-world deployments. We propose FLARE, an adaptive reputation-based framework that transforms client reliability assessment from binary decisions to a continuous, multi-dimensional trust evaluation. FLARE integrates: (i) a multi-dimensional reputation score capturing performance consistency, statistical anomaly indicators, and temporal behavior, (ii) a self-calibrating adaptive threshold mechanism that adjusts security strictness based on model convergence and recent attack intensity, (iii) reputation-weighted aggregation with soft exclusion to proportionally limit suspicious contributions rather than eliminating clients outright, and (iv) a Local Differential Privacy (LDP) mechanism enabling reputation scoring on privatized client updates. We further introduce a highly evasive Statistical Mimicry (SM) attack, a benchmark adversary that blends honest gradients with synthetic perturbations and persistent drift to remain undetected by traditional filters. Extensive experiments with 100 clients on MNIST, CIFAR-10, and SVHN demonstrate that FLARE maintains high model accuracy and converges faster than state-of-the-art Byzantine-robust methods under diverse attack types, including label flipping, gradient scaling, adaptive attacks, ALIE, and SM. FLARE improves robustness by up to 16% and preserves model convergence within 30% of the non-attacked baseline, while achieving strong malicious-client detection performance with minimal computational overhead. https://github.com/Anonymous0-0paper/FLARE

摘要: 联合学习（FL）支持协作模型训练，同时保护数据隐私。然而，它仍然容易受到恶意客户的攻击，这些客户通过拜占庭攻击、数据中毒或适应性对抗行为损害模型完整性。现有的防御机制依赖于静态阈值和二进制分类，无法适应现实世界部署中不断变化的客户端行为。我们提出了FLARE，一个自适应的声誉为基础的框架，将客户端的可靠性评估从二进制的决定，一个连续的，多维的信任评估。FLARE集成了：（i）捕获性能一致性、统计异常指标和时间行为的多维信誉分数，（ii）基于模型收敛和最近攻击强度来调整安全严格性的自校准自适应阈值机制，（iii）具有软排除的信誉加权聚合，以按比例限制可疑贡献而不是彻底消除客户端，以及（iv）局部差异隐私（LDP）机制，其使得能够对私有化的客户端更新进行信誉评分。我们进一步引入了一种高度规避的统计模仿（SM）攻击，这是一种基准对手，将诚实梯度与合成扰动和持续漂移相结合，以保持传统过滤器无法检测到。在MNIST、CIFAR-10和SVHN上对100个客户端进行的广泛实验表明，在多种攻击类型（包括标签翻转、梯度缩放、自适应攻击、ALIE和SM）下，LGA保持了高模型准确性，并且比最先进的拜占庭鲁棒方法收敛得更快。LGA将稳健性提高了高达16%，并将模型收敛性保持在未受攻击基线的30%以内，同时以最小的计算负担实现强大的恶意客户端检测性能。https://github.com/Anonymous0-0paper/FLARE



## **27. Sigil: Server-Enforced Watermarking in U-Shaped Split Federated Learning via Gradient Injection**

Sigil：通过梯度注入在U形拆分联邦学习中实现服务器强制水印 cs.CR

18 pages,8 figures

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14422v1) [paper-pdf](https://arxiv.org/pdf/2511.14422v1)

**Authors**: Zhengchunmin Dai, Jiaxiong Tang, Peng Sun, Honglong Chen, Liantao Wu

**Abstract**: In decentralized machine learning paradigms such as Split Federated Learning (SFL) and its variant U-shaped SFL, the server's capabilities are severely restricted. Although this enhances client-side privacy, it also leaves the server highly vulnerable to model theft by malicious clients. Ensuring intellectual property protection for such capability-limited servers presents a dual challenge: watermarking schemes that depend on client cooperation are unreliable in adversarial settings, whereas traditional server-side watermarking schemes are technically infeasible because the server lacks access to critical elements such as model parameters or labels.   To address this challenge, this paper proposes Sigil, a mandatory watermarking framework designed specifically for capability-limited servers. Sigil defines the watermark as a statistical constraint on the server-visible activation space and embeds the watermark into the client model via gradient injection, without requiring any knowledge of the data. Besides, we design an adaptive gradient clipping mechanism to ensure that our watermarking process remains both mandatory and stealthy, effectively countering existing gradient anomaly detection methods and a specifically designed adaptive subspace removal attack. Extensive experiments on multiple datasets and models demonstrate Sigil's fidelity, robustness, and stealthiness.

摘要: 在分散式机器学习范式（例如拆分联邦学习（SFL）及其变体U形SFL）中，服务器的能力受到严格限制。尽管这增强了客户端隐私，但也使服务器极易受到恶意客户端的模型窃取的影响。确保此类能力有限的服务器的知识产权保护提出了双重挑战：依赖于客户端合作的水印方案在对抗性环境中是不可靠的，而传统的服务器端水印方案在技术上是不可行的，因为服务器缺乏对模型参数或标签等关键元素的访问。   为了应对这一挑战，本文提出了Sigil，这是一个专门为能力有限的服务器设计的强制水印框架。Sigil将水印定义为服务器可见激活空间上的统计约束，并通过梯度注入将水印嵌入到客户端模型中，而不需要任何数据知识。此外，我们设计了一种自适应梯度剪裁机制，以确保我们的水印过程保持强制性和隐蔽性，有效地对抗现有的梯度异常检测方法和专门设计的自适应子空间去除攻击。对多个数据集和模型的广泛实验证明了Sigil的保真度、稳健性和隐蔽性。



## **28. Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving**

全规模作弊立体匹配：自动驾驶中针对双眼深度估计的物理对抗攻击 cs.CV

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.14386v2) [paper-pdf](https://arxiv.org/pdf/2511.14386v2)

**Authors**: Kangqiao Zhao, Shuo Huai, Xurui Song, Jun Luo

**Abstract**: Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.

摘要: 尽管用于实现自动驾驶感知的深度神经模型已被证明容易受到对抗示例的影响，但已知的攻击通常利用2D补丁并主要针对单目感知。因此，物理对抗示例（PAEs）对基于立体声的双眼深度估计的有效性在很大程度上仍然没有被探索。为此，我们在自动驾驶背景下提出了针对立体匹配模型的第一个基于纹理的物理对抗攻击。我们的方法采用具有全局伪装纹理的3D PCE，而不是基于局部2D补丁的纹理，确保立体相机不同视角的视觉一致性和攻击有效性。为了应对这些摄像机的差异效应，我们还提出了一种新的3D立体匹配渲染模块，该模块允许PCE与双眼视觉中的真实位置和航向对齐。我们进一步提出了一种新颖的合并攻击，通过细粒度的PCE优化将目标无缝地混合到环境中。它显着增强了对现有无法无缝合并到后台的隐藏攻击的隐蔽性和杀伤力。广泛的评估表明，我们的PEN可以成功地欺骗立体模型产生错误的深度信息。



## **29. Steganographic Backdoor Attacks in NLP: Ultra-Low Poisoning and Defense Evasion**

NLP中的隐写后门攻击：超低中毒和防御逃避 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14301v1) [paper-pdf](https://arxiv.org/pdf/2511.14301v1)

**Authors**: Eric Xue, Ruiyi Zhang, Zijun Zhang, Pengtao Xie

**Abstract**: Transformer models are foundational to natural language processing (NLP) applications, yet remain vulnerable to backdoor attacks introduced through poisoned data, which implant hidden behaviors during training. To strengthen the ability to prevent such compromises, recent research has focused on designing increasingly stealthy attacks to stress-test existing defenses, pairing backdoor behaviors with stylized artifact or token-level perturbation triggers. However, this trend diverts attention from the harder and more realistic case: making the model respond to semantic triggers such as specific names or entities, where a successful backdoor could manipulate outputs tied to real people or events in deployed systems. Motivated by this growing disconnect, we introduce SteganoBackdoor, bringing stealth techniques back into line with practical threat models. Leveraging innocuous properties from natural-language steganography, SteganoBackdoor applies a gradient-guided data optimization process to transform semantic trigger seeds into steganographic carriers that embed a high backdoor payload, remain fluent, and exhibit no representational resemblance to the trigger. Across diverse experimental settings, SteganoBackdoor achieves over 99% attack success at an order-of-magnitude lower data-poisoning rate than prior approaches while maintaining unparalleled evasion against a comprehensive suite of data-level defenses. By revealing this practical and covert attack, SteganoBackdoor highlights an urgent blind spot in current defenses and demands immediate attention to adversarial data defenses and real-world threat modeling.

摘要: Transformer模型是自然语言处理（NLP）应用程序的基础，但仍然容易受到通过有毒数据引入的后门攻击，这些攻击在训练期间植入隐藏行为。为了加强防止此类妥协的能力，最近的研究重点是设计越来越隐蔽的攻击来压力测试现有的防御，将后门行为与风格化的文物或代币级扰动触发器配对。然而，这种趋势转移了人们对更困难、更现实情况的注意力：让模型响应特定名称或实体等语义触发器，其中成功的后门可以操纵与部署的系统中的真实人或事件相关的输出。受这种日益严重的脱节的激励，我们引入了SteganoBackdoor，使隐形技术与实际威胁模型重新保持一致。利用自然语言隐写术的无害属性，SteganoBackdoor应用梯度引导的数据优化过程，将语义触发种子转换为隐写载体，嵌入高后门有效载荷，保持流畅，并且没有表现出与触发器的代表性相似。在不同的实验环境中，SteganoBackdoor以比以前方法低一个数量级的数据中毒率实现了超过99%的攻击成功率，同时保持了对一套全面的数据级防御的无与伦比的规避。通过揭示这种实际和隐蔽的攻击，SteganoBackdoor强调了当前防御中的一个紧迫盲点，并要求立即关注对抗性数据防御和现实世界的威胁建模。



## **30. A Fuzzy Logic-Based Cryptographic Framework For Real-Time Dynamic Key Generation For Enhanced Data Encryption**

用于增强数据加密的实时动态密钥生成的基于模糊逻辑的加密框架 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14132v1) [paper-pdf](https://arxiv.org/pdf/2511.14132v1)

**Authors**: Kavya Bhand, Payal Khubchandani, Jyoti Khubchandani

**Abstract**: With the ever-growing demand for cybersecurity, static key encryption mechanisms are increasingly vulnerable to adversarial attacks due to their deterministic and non-adaptive nature. Brute-force attacks, key compromise, and unauthorized access have become highly common cyber threats. This research presents a novel fuzzy logic-based cryptographic framework that dynamically generates encryption keys in real-time by accessing system-level entropy and hardware-bound trust. The proposed system leverages a Fuzzy Inference System (FIS) to evaluate system parameters that include CPU utilization, process count, and timestamp variation. It assigns entropy level based on linguistically defined fuzzy rules which are fused with hardware-generated randomness and then securely sealed using a Trusted Platform Module (TPM). The sealed key is incorporated in an AES-GCM encryption scheme to ensure both confidentiality and integrity of the data. This system introduces a scalable solution for adaptive encryption in high-assurance computing, zero-trust environments, and cloud-based infrastructure.

摘要: 随着对网络安全的需求不断增长，静态密钥加密机制由于其确定性和非适应性，越来越容易受到对抗攻击。暴力攻击、密钥泄露和未经授权的访问已成为非常常见的网络威胁。这项研究提出了一种新型的基于模糊逻辑的加密框架，该框架通过访问系统级的信息量和硬件绑定的信任度来实时动态生成加密密钥。所提出的系统利用模糊推理系统（FIS）来评估系统参数，包括中央处理器利用率、进程计数和时间戳变化。它基于语言定义的模糊规则来分配熵级别，这些规则与硬件生成的随机性融合，然后使用可信协议安全地密封。密封的密钥包含在AES-GCM加密方案中，以确保数据的机密性和完整性。该系统引入了高保证计算、零信任环境和基于云的基础设施中的自适应加密的可扩展解决方案。



## **31. Dynamic Black-box Backdoor Attacks on IoT Sensory Data**

物联网感知数据的动态黑盒后门攻击 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14074v1) [paper-pdf](https://arxiv.org/pdf/2511.14074v1)

**Authors**: Ajesh Koyatan Chathoth, Stephen Lee

**Abstract**: Sensor data-based recognition systems are widely used in various applications, such as gait-based authentication and human activity recognition (HAR). Modern wearable and smart devices feature various built-in Inertial Measurement Unit (IMU) sensors, and such sensor-based measurements can be fed to a machine learning-based model to train and classify human activities. While deep learning-based models have proven successful in classifying human activity and gestures, they pose various security risks. In our paper, we discuss a novel dynamic trigger-generation technique for performing black-box adversarial attacks on sensor data-based IoT systems. Our empirical analysis shows that the attack is successful on various datasets and classifier models with minimal perturbation on the input data. We also provide a detailed comparative analysis of performance and stealthiness to various other poisoning techniques found in backdoor attacks. We also discuss some adversarial defense mechanisms and their impact on the effectiveness of our trigger-generation technique.

摘要: 基于传感器数据的识别系统广泛用于各种应用，例如基于步态的认证和人类活动识别（HAR）。现代可穿戴和智能设备具有各种内置惯性测量单元（IMU）传感器，此类基于传感器的测量可以被反馈到基于机器学习的模型来训练和分类人类活动。虽然基于深度学习的模型已被证明在对人类活动和手势进行分类方面是成功的，但它们也带来了各种安全风险。在我们的论文中，我们讨论了一种新型的动态命令生成技术，用于对基于传感器数据的物联网系统执行黑匣子对抗攻击。我们的实证分析表明，攻击对各种数据集和分类器模型都是成功的，并且对输入数据的干扰最小。我们还对后门攻击中发现的各种其他中毒技术的性能和隐蔽性进行了详细的比较分析。我们还讨论了一些对抗性防御机制及其对我们的士兵生成技术有效性的影响。



## **32. Accuracy is Not Enough: Poisoning Interpretability in Federated Learning via Color Skew**

准确性还不够：颜色倾斜毒害联邦学习中的解释性 cs.CV

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.13535v2) [paper-pdf](https://arxiv.org/pdf/2511.13535v2)

**Authors**: Farhin Farhad Riya, Shahinul Hoque, Jinyuan Stella Sun, Olivera Kotevska

**Abstract**: As machine learning models are increasingly deployed in safety-critical domains, visual explanation techniques have become essential tools for supporting transparency. In this work, we reveal a new class of attacks that compromise model interpretability without affecting accuracy. Specifically, we show that small color perturbations applied by adversarial clients in a federated learning setting can shift a model's saliency maps away from semantically meaningful regions while keeping the prediction unchanged. The proposed saliency-aware attack framework, called Chromatic Perturbation Module, systematically crafts adversarial examples by altering the color contrast between foreground and background in a way that disrupts explanation fidelity. These perturbations accumulate across training rounds, poisoning the global model's internal feature attributions in a stealthy and persistent manner. Our findings challenge a common assumption in model auditing that correct predictions imply faithful explanations and demonstrate that interpretability itself can be an attack surface. We evaluate this vulnerability across multiple datasets and show that standard training pipelines are insufficient to detect or mitigate explanation degradation, especially in the federated learning setting, where subtle color perturbations are harder to discern. Our attack reduces peak activation overlap in Grad-CAM explanations by up to 35% while preserving classification accuracy above 96% on all evaluated datasets.

摘要: 随着机器学习模型越来越多地部署在安全关键领域，视觉解释技术已成为支持透明度的重要工具。在这项工作中，我们揭示了一类新型攻击，这些攻击在不影响准确性的情况下损害了模型的可解释性。具体来说，我们表明，在联邦学习环境中，对抗客户端应用的小颜色扰动可以将模型的显着性地图从语义有意义的区域移开，同时保持预测不变。提出的显着性感知攻击框架称为色彩扰动模块，通过以破坏解释保真度的方式改变前景和背景之间的颜色对比度，系统性地制作对抗性示例。这些扰动在训练轮中累积，以一种隐秘且持续的方式毒害了全球模型的内部特征属性。我们的发现挑战了模型审计中的一个常见假设，即正确的预测意味着忠实的解释，并证明可解释性本身可能是一种攻击面。我们在多个数据集中评估了这个漏洞，并表明标准训练管道不足以检测或减轻解释退化，特别是在联邦学习环境中，微妙的颜色扰动更难辨别。我们的攻击将Grad-CAM解释中的峰值激活重叠减少了高达35%，同时在所有评估的数据集上将分类准确率保持在96%以上。



## **33. MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization**

MPD-SGR：膜电位分布驱动的替代梯度正则化鲁棒脉冲神经网络 cs.LG

Accepted by AAAI 2026

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.12199v2) [paper-pdf](https://arxiv.org/pdf/2511.12199v2)

**Authors**: Runhao Jiang, Chengzhi Jiang, Rui Yan, Huajin Tang

**Abstract**: The surrogate gradient (SG) method has shown significant promise in enhancing the performance of deep spiking neural networks (SNNs), but it also introduces vulnerabilities to adversarial attacks. Although spike coding strategies and neural dynamics parameters have been extensively studied for their impact on robustness, the critical role of gradient magnitude, which reflects the model's sensitivity to input perturbations, remains underexplored. In SNNs, the gradient magnitude is primarily determined by the interaction between the membrane potential distribution (MPD) and the SG function. In this study, we investigate the relationship between the MPD and SG and their implications for improving the robustness of SNNs. Our theoretical analysis reveals that reducing the proportion of membrane potentials lying within the gradient-available range of the SG function effectively mitigates the sensitivity of SNNs to input perturbations. Building upon this insight, we propose a novel MPD-driven surrogate gradient regularization (MPD-SGR) method, which enhances robustness by explicitly regularizing the MPD based on its interaction with the SG function. Extensive experiments across multiple image classification benchmarks and diverse network architectures confirm that the MPD-SGR method significantly enhances the resilience of SNNs to adversarial perturbations and exhibits strong generalizability across diverse network configurations, SG functions, and spike encoding schemes.

摘要: 代理梯度（SG）方法在增强深度尖峰神经网络（SNN）的性能方面表现出了巨大的希望，但它也引入了对抗性攻击的漏洞。尽管尖峰编码策略和神经动力学参数对鲁棒性的影响已被广泛研究，但反映模型对输入扰动敏感性的梯度幅度的关键作用仍然没有得到充分研究。在SNN中，梯度大小主要由膜势分布（CPD）和SG函数之间的相互作用决定。在这项研究中，我们研究了CPD和SG之间的关系及其对提高SNN稳健性的影响。我们的理论分析表明，减少位于SG函数的梯度可用范围内的膜势比例可以有效地降低SNN对输入扰动的敏感性。基于这一见解，我们提出了一种新型的CPD驱动的代理梯度正规化（MPD-SGR）方法，该方法通过根据其与SG函数的相互作用显式正规化MPD来增强鲁棒性。跨多个图像分类基准和不同网络架构的大量实验证实，MPD-SGR方法显着增强了SNN对对抗性扰动的弹性，并在不同网络配置、SG功能和尖峰编码方案中表现出强大的通用性。



## **34. Privacy on the Fly: A Predictive Adversarial Transformation Network for Mobile Sensor Data**

动态隐私：移动传感器数据的预测性对抗转换网络 cs.CR

accepted by AAAI 2026 (oral)

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.07242v3) [paper-pdf](https://arxiv.org/pdf/2511.07242v3)

**Authors**: Tianle Song, Chenhao Lin, Yang Cao, Zhengyu Zhao, Jiahao Sun, Chong Zhang, Le Yang, Chao Shen

**Abstract**: Mobile motion sensors such as accelerometers and gyroscopes are now ubiquitously accessible by third-party apps via standard APIs. While enabling rich functionalities like activity recognition and step counting, this openness has also enabled unregulated inference of sensitive user traits, such as gender, age, and even identity, without user consent. Existing privacy-preserving techniques, such as GAN-based obfuscation or differential privacy, typically require access to the full input sequence, introducing latency that is incompatible with real-time scenarios. Worse, they tend to distort temporal and semantic patterns, degrading the utility of the data for benign tasks like activity recognition. To address these limitations, we propose the Predictive Adversarial Transformation Network (PATN), a real-time privacy-preserving framework that leverages historical signals to generate adversarial perturbations proactively. The perturbations are applied immediately upon data acquisition, enabling continuous protection without disrupting application functionality. Experiments on two datasets demonstrate that PATN substantially degrades the performance of privacy inference models, achieving Attack Success Rate (ASR) of 40.11% and 44.65% (reducing inference accuracy to near-random) and increasing the Equal Error Rate (EER) from 8.30% and 7.56% to 41.65% and 46.22%. On ASR, PATN outperforms baseline methods by 16.16% and 31.96%, respectively.

摘要: 现在，第三方应用程序通过标准API无处不在地访问加速度计和陀螺仪等移动运动传感器。这种开放性在实现活动识别和步数等丰富功能的同时，还可以在未经用户同意的情况下对敏感的用户特征（例如性别、年龄甚至身份）进行不受监管的推断。现有的隐私保护技术，例如基于GAN的模糊或差异隐私，通常需要访问完整的输入序列，从而引入与实时场景不兼容的延迟。更糟糕的是，它们往往会扭曲时间和语义模式，降低数据对活动识别等良性任务的实用性。为了解决这些限制，我们提出了预测对抗转换网络（PATN），这是一个实时隐私保护框架，可以利用历史信号主动生成对抗性扰动。数据采集后立即应用扰动，从而实现持续保护，而不会中断应用程序功能。对两个数据集的实验表明，PATN大幅降低了隐私推理模型的性能，实现了40.11%和44.65%的攻击成功率（ASB）（将推理准确率降低到接近随机），并将等错误率（EER）从8.30%和7.56%提高到41.65%和46.22%。在ASB方面，PATN分别比基线方法高出16.16%和31.96%。



## **35. Injecting Falsehoods: Adversarial Man-in-the-Middle Attacks Undermining Factual Recall in LLMs**

注入谎言：对抗性中间人攻击破坏LLM中的事实召回 cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.05919v2) [paper-pdf](https://arxiv.org/pdf/2511.05919v2)

**Authors**: Alina Fastowski, Bardh Prenkaj, Yuxiao Li, Gjergji Kasneci

**Abstract**: LLMs are now an integral part of information retrieval. As such, their role as question answering chatbots raises significant concerns due to their shown vulnerability to adversarial man-in-the-middle (MitM) attacks. Here, we propose the first principled attack evaluation on LLM factual memory under prompt injection via Xmera, our novel, theory-grounded MitM framework. By perturbing the input given to "victim" LLMs in three closed-book and fact-based QA settings, we undermine the correctness of the responses and assess the uncertainty of their generation process. Surprisingly, trivial instruction-based attacks report the highest success rate (up to ~85.3%) while simultaneously having a high uncertainty for incorrectly answered questions. To provide a simple defense mechanism against Xmera, we train Random Forest classifiers on the response uncertainty levels to distinguish between attacked and unattacked queries (average AUC of up to ~96%). We believe that signaling users to be cautious about the answers they receive from black-box and potentially corrupt LLMs is a first checkpoint toward user cyberspace safety.

摘要: LLM现在是信息检索不可或缺的一部分。因此，它们作为问答聊天机器人的角色引起了严重的担忧，因为它们表现出容易受到对抗性中间人（MitM）攻击的脆弱性。在这里，我们提出了在通过Xmera（我们新颖的、基于理论的MitM框架）即时注入下对LLM事实记忆的第一个原则性攻击评估。通过扰乱三种闭门和基于事实的QA环境中提供给“受害者”LLM的输入，我们破坏了响应的正确性并评估其生成过程的不确定性。令人惊讶的是，基于描述的琐碎攻击的成功率最高（高达~85.3%），同时对错误回答的问题具有很高的不确定性。为了提供针对Xmera的简单防御机制，我们在响应不确定性水平上训练随机森林分类器，以区分受攻击和未受攻击的查询（平均AUDA高达~96%）。我们相信，向用户发出信号，要求用户对从黑匣子和潜在腐败的LLM收到的答案保持谨慎，是用户网络空间安全的第一个检查站。



## **36. DRIP: Defending Prompt Injection via Token-wise Representation Editing and Residual Instruction Fusion**

DRIP：通过逐令牌表示编辑和剩余指令融合来防御提示注入 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.00447v2) [paper-pdf](https://arxiv.org/pdf/2511.00447v2)

**Authors**: Ruofan Liu, Yun Lin, Zhiyong Huang, Jin Song Dong

**Abstract**: Large language models (LLMs) are increasingly integrated into IT infrastructures, where they process user data according to predefined instructions. However, conventional LLMs remain vulnerable to prompt injection, where malicious users inject directive tokens into the data to subvert model behavior. Existing defenses train LLMs to semantically separate data and instruction tokens, but still struggle to (1) balance utility and security and (2) prevent instruction-like semantics in the data from overriding the intended instructions.   We propose DRIP, which (1) precisely removes instruction semantics from tokens in the data section while preserving their data semantics, and (2) robustly preserves the effect of the intended instruction even under strong adversarial content. To "de-instructionalize" data tokens, DRIP introduces a data curation and training paradigm with a lightweight representation-editing module that edits embeddings of instruction-like tokens in the data section, enhancing security without harming utility. To ensure non-overwritability of instructions, DRIP adds a minimal residual module that reduces the ability of adversarial data to overwrite the original instruction. We evaluate DRIP on LLaMA 8B and Mistral 7B against StruQ, SecAlign, ISE, and PFT on three prompt-injection benchmarks (SEP, AlpacaFarm, and InjecAgent). DRIP improves role-separation score by 12-49\%, reduces attack success rate by over 66\% under adaptive attacks, and matches the utility of the undefended model, establishing a new state of the art for prompt-injection robustness.

摘要: 大型语言模型（LLM）越来越多地集成到IT基础设施中，它们根据预定义的指令处理用户数据。然而，传统的LLM仍然容易受到提示注入的影响，即恶意用户将指令令牌注入到数据中以颠覆模型行为。现有的防御措施训练LLM在语义上分离数据和指令令牌，但仍然难以（1）平衡实用性和安全性，以及（2）防止数据中类似描述的语义覆盖预期指令。   我们提出了DRIP，它（1）从数据部分中的令牌中精确地删除指令语义，同时保留其数据语义，（2）即使在强对抗性内容下也能稳健地保留预期指令的效果。为了“去伪化”数据令牌，DRIP引入了一种数据策展和训练范式，该范式具有轻量级的表示编辑模块，该模块可以编辑数据部分中类似描述的令牌的嵌入，从而在不损害实用性的情况下增强了安全性。为了确保指令的不可重写性，DRIP添加了一个最小剩余模块，该模块降低了对抗数据重写原始指令的能力。我们在三个预算注入基准（SDP、AlpacaFarm和InjecAgent）上针对StruQ、SecAlign、ISE和PFT评估了LLaMA 8B和Mistral 7 B上的DRIP。DRIP将角色分离分数提高了12- 49%，在适应性攻击下将攻击成功率降低了66%以上，并匹配了无防御模型的实用性，为预算注入鲁棒性建立了新的最新水平。



## **37. Observation-Free Attacks on Online Learning to Rank**

对在线学习排名的无观察攻击 cs.LG

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2509.22855v3) [paper-pdf](https://arxiv.org/pdf/2509.22855v3)

**Authors**: Sameep Chattopadhyay, Nikhil Karamchandani, Sharayu Moharir

**Abstract**: Online learning to rank (OLTR) plays a critical role in information retrieval and machine learning systems, with a wide range of applications in search engines and content recommenders. However, despite their extensive adoption, the susceptibility of OLTR algorithms to coordinated adversarial attacks remains poorly understood. In this work, we present a novel framework for attacking some of the widely used OLTR algorithms. Our framework is designed to promote a set of target items so that they appear in the list of top-K recommendations for T - o(T) rounds, while simultaneously inducing linear regret in the learning algorithm. We propose two novel attack strategies: CascadeOFA for CascadeUCB1 and PBMOFA for PBM-UCB . We provide theoretical guarantees showing that both strategies require only O(log T) manipulations to succeed. Additionally, we supplement our theoretical analysis with empirical results on real-world data.

摘要: 在线排名学习（OLTR）在信息检索和机器学习系统中发挥着至关重要的作用，在搜索引擎和内容排序器中有广泛的应用。然而，尽管OLTR算法被广泛采用，但人们对OLTR算法对协同对抗攻击的敏感性仍然知之甚少。在这项工作中，我们提出了一个新颖的框架来攻击一些广泛使用的OLTR算法。我们的框架旨在推广一组目标项，使它们出现在T-o（T）轮的前K推荐列表中，同时在学习算法中引发线性遗憾。我们提出了两种新颖的攻击策略：针对CascadeUCB 1的CascadeOFA和针对PBM-UCB的PBMOFA。我们提供了理论保证，表明这两种策略只需要O（log T）操作即可成功。此外，我们还通过现实世界数据的实证结果来补充理论分析。



## **38. Decoding Deception: Understanding Automatic Speech Recognition Vulnerabilities in Evasion and Poisoning Attacks**

解码欺骗：了解逃避和中毒攻击中的自动语音识别漏洞 cs.SD

Remove due to conflict in authors

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2509.22060v2) [paper-pdf](https://arxiv.org/pdf/2509.22060v2)

**Authors**: Aravindhan G, Yuvaraj Govindarajulu, Parin Shah

**Abstract**: Recent studies have demonstrated the vulnerability of Automatic Speech Recognition systems to adversarial examples, which can deceive these systems into misinterpreting input speech commands. While previous research has primarily focused on white-box attacks with constrained optimizations, and transferability based black-box attacks against commercial Automatic Speech Recognition devices, this paper explores cost efficient white-box attack and non transferability black-box adversarial attacks on Automatic Speech Recognition systems, drawing insights from approaches such as Fast Gradient Sign Method and Zeroth-Order Optimization. Further, the novelty of the paper includes how poisoning attack can degrade the performances of state-of-the-art models leading to misinterpretation of audio signals. Through experimentation and analysis, we illustrate how hybrid models can generate subtle yet impactful adversarial examples with very little perturbation having Signal Noise Ratio of 35dB that can be generated within a minute. These vulnerabilities of state-of-the-art open source model have practical security implications, and emphasize the need for adversarial security.

摘要: 最近的研究表明，自动语音识别系统对对抗示例的脆弱性，这些示例可能会欺骗这些系统误解输入语音命令。虽然之前的研究主要集中在具有约束优化的白盒攻击以及针对商用自动语音识别设备的基于可转移性的黑匣子攻击，但本文探索了对自动语音识别系统的成本高效白盒攻击和不可转移性的黑匣子对抗攻击，从快速梯度符号法和零阶优化等方法中汲取见解。此外，该论文的新颖性包括中毒攻击如何降低最先进模型的性能，从而导致音频信号的误解。通过实验和分析，我们说明了混合模型如何在很小的干扰下生成微妙但有影响力的对抗示例，可以在一分钟内生成具有35分贝的信噪比。最先进的开源模型的这些漏洞具有实际的安全影响，并强调了对抗性安全的需要。



## **39. SoK: Exposing the Generation and Detection Gaps in LLM-Generated Phishing Through Examination of Generation Methods, Content Characteristics, and Countermeasures**

SoK：通过检查生成方法、内容特征和对策来暴露LLM生成的网络钓鱼中的生成和检测差距 cs.CR

18 pages, 5 tables, 4 figures

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2508.21457v2) [paper-pdf](https://arxiv.org/pdf/2508.21457v2)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Phishing campaigns involve adversaries masquerading as trusted vendors trying to trigger user behavior that enables them to exfiltrate private data. While URLs are an important part of phishing campaigns, communicative elements like text and images are central in triggering the required user behavior. Further, due to advances in phishing detection, attackers react by scaling campaigns to larger numbers and diversifying and personalizing content. In addition to established mechanisms, such as template-based generation, large language models (LLMs) can be used for phishing content generation, enabling attacks to scale in minutes, challenging existing phishing detection paradigms through personalized content, stealthy explicit phishing keywords, and dynamic adaptation to diverse attack scenarios. Countering these dynamically changing attack campaigns requires a comprehensive understanding of the complex LLM-related threat landscape. Existing studies are fragmented and focus on specific areas. In this work, we provide the first holistic examination of LLM-generated phishing content. First, to trace the exploitation pathways of LLMs for phishing content generation, we adopt a modular taxonomy documenting nine stages by which adversaries breach LLM safety guardrails. We then characterize how LLM-generated phishing manifests as threats, revealing that it evades detectors while emphasizing human cognitive manipulation. Third, by taxonomizing defense techniques aligned with generation methods, we expose a critical asymmetry that offensive mechanisms adapt dynamically to attack scenarios, whereas defensive strategies remain static and reactive. Finally, based on a thorough analysis of the existing literature, we highlight insights and gaps and suggest a roadmap for understanding and countering LLM-driven phishing at scale.

摘要: 网络钓鱼活动涉及伪装成值得信赖的供应商的对手，试图触发用户行为，使他们能够泄露私人数据。虽然URL是网络钓鱼活动的重要组成部分，但文本和图像等通信元素是触发所需用户行为的核心。此外，由于网络钓鱼检测的进步，攻击者通过将活动规模扩大到更大的数量以及使内容多样化和个性化来做出反应。除了基于模板的生成等已建立的机制外，大型语言模型（LLM）还可用于网络钓鱼内容生成，使攻击在几分钟内扩展，通过个性化内容、隐形显式网络钓鱼关键词和动态适应各种攻击场景来挑战现有的网络钓鱼检测范式。应对这些动态变化的攻击活动需要全面了解复杂的LLM相关威胁格局。现有的研究是零散的，并且集中在特定领域。在这项工作中，我们对LLM生成的网络钓鱼内容进行了首次全面检查。首先，为了追踪LLM用于网络钓鱼内容生成的利用途径，我们采用模块化分类法，记录对手突破LLM安全护栏的九个阶段。然后，我们描述了LLM生成的网络钓鱼如何表现为威胁，揭示了它可以逃避检测器，同时强调人类认知操纵。第三，通过对与生成方法保持一致的防御技术进行分类，我们暴露了一个关键的不对称性，即攻击机制动态地适应攻击场景，而防御策略保持静态和反应性。最后，根据对现有文献的彻底分析，我们强调了见解和差距，并提出了大规模理解和打击LLM驱动的网络钓鱼的路线图。



## **40. Constraint-Guided Prediction Refinement via Deterministic Diffusion Trajectories**

通过确定性扩散轨迹的约束引导预测细化 cs.AI

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2506.12911v2) [paper-pdf](https://arxiv.org/pdf/2506.12911v2)

**Authors**: Pantelis Dogoulis, Fabien Bernier, Félix Fourreau, Karim Tit, Maxime Cordy

**Abstract**: Many real-world machine learning tasks require outputs that satisfy hard constraints, such as physical conservation laws, structured dependencies in graphs, or column-level relationships in tabular data. Existing approaches rely either on domain-specific architectures and losses or on strong assumptions on the constraint space, restricting their applicability to linear or convex constraints. We propose a general-purpose framework for constraint-aware refinement that leverages denoising diffusion implicit models (DDIMs). Starting from a coarse prediction, our method iteratively refines it through a deterministic diffusion trajectory guided by a learned prior and augmented by constraint gradient corrections. The approach accommodates a wide class of non-convex and nonlinear equality constraints and can be applied post hoc to any base model. We demonstrate the method in two representative domains: constrained adversarial attack generation on tabular data with column-level dependencies and in AC power flow prediction under Kirchhoff's laws. Across both settings, our diffusion-guided refinement improves both constraint satisfaction and performance while remaining lightweight and model-agnostic.

摘要: 许多现实世界的机器学习任务需要满足硬约束的输出，例如物理保守定律、图形中的结构化依赖关系或表格数据中的列级关系。现有的方法要么依赖于特定领域的架构和损失，要么依赖于对约束空间的强假设，从而将其适用性限制在线性或凸约束中。我们提出了一个用于约束感知细化的通用框架，该框架利用去噪扩散隐式模型（DDIM）。我们的方法从粗略预测开始，通过由学习先验引导并通过约束梯度修正增强的确定性扩散轨迹来迭代细化它。该方法适应广泛的非凸和非线性等式约束，并且可以事后应用于任何基本模型。我们在两个代表性领域展示了该方法：对具有列级依赖关系的表格数据产生约束对抗攻击，以及根据基尔霍夫定律进行交流潮流预测。在这两种设置中，我们的扩散引导细化可以提高约束满意度和性能，同时保持轻量级和模型不可知。



## **41. TooBadRL: Trigger Optimization to Boost Effectiveness of Backdoor Attacks on Deep Reinforcement Learning**

TooBadRL：触发优化以提高后门攻击对深度强化学习的有效性 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2506.09562v3) [paper-pdf](https://arxiv.org/pdf/2506.09562v3)

**Authors**: Mingxuan Zhang, Oubo Ma, Kang Wei, Songze Li, Shouling Ji

**Abstract**: Deep reinforcement learning (DRL) has achieved remarkable success in a wide range of sequential decision-making applications, including robotics, healthcare, smart grids, and finance. Recent studies reveal that adversaries can implant backdoors into DRL agents during the training phase. These backdoors can later be activated by specific triggers during deployment, compelling the agent to execute targeted actions and potentially leading to severe consequences, such as drone crashes or vehicle collisions. However, existing backdoor attacks utilize simplistic and heuristic trigger configurations, overlooking the critical impact of trigger design on attack effectiveness. To address this gap, we introduce TooBadRL, the first framework to systematically optimize DRL backdoor triggers across three critical aspects: injection timing, trigger dimension, and manipulation magnitude. Specifically, we first introduce a performance-aware adaptive freezing mechanism to determine the injection timing during training. Then, we formulate trigger selection as an influence attribution problem and apply Shapley value analysis to identify the most influential trigger dimension for injection. Furthermore, we propose an adversarial input synthesis method to optimize the manipulation magnitude under environmental constraints. Extensive evaluations on three DRL algorithms and nine benchmark tasks demonstrate that TooBadRL outperforms five baseline methods in terms of attack success rate while only slightly affecting normal task performance. We further evaluate potential defense strategies from detection and mitigation perspectives. We open-source our code to facilitate reproducibility and further research.

摘要: 深度强化学习（DRL）在机器人、医疗保健、智能电网和金融等广泛的顺序决策应用中取得了显着的成功。最近的研究表明，对手可以在训练阶段向DRL特工植入后门。这些后门后来可能会在部署期间被特定触发器激活，迫使代理执行有针对性的操作，并可能导致严重后果，例如无人机坠毁或车辆碰撞。然而，现有的后门攻击利用简单和启发式的触发配置，忽视了触发设计对攻击有效性的关键影响。为了解决这一差距，我们引入了TooBadRL，这是第一个在三个关键方面系统优化DRL后门触发器的框架：注入时间，触发维度和操纵幅度。具体来说，我们首先引入了一个性能感知的自适应冻结机制，以确定在训练过程中的注入时间。然后，我们制定了触发器选择的影响归因问题，并应用Shapley值分析，以确定最有影响力的触发器尺寸注入。此外，我们提出了一种对抗性输入合成方法，以优化环境约束下的操作幅度。对三种DRL算法和九个基准任务的广泛评估表明，TooBadRL在攻击成功率方面优于五种基线方法，而对正常任务性能的影响很小。我们进一步从检测和缓解的角度评估潜在的防御策略。我们开源我们的代码，以促进可重复性和进一步的研究。



## **42. Revisiting Model Inversion Evaluation: From Misleading Standards to Reliable Privacy Assessment**

重温模型倒置评估：从误导性标准到可靠的隐私评估 cs.LG

To support future work, we release our MLLM-based MI evaluation framework and benchmarking suite at https://github.com/hosytuyen/MI-Eval-MLLM

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2505.03519v4) [paper-pdf](https://arxiv.org/pdf/2505.03519v4)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information from private training data by exploiting access to machine learning models T. To evaluate such attacks, the standard evaluation framework relies on an evaluation model E, trained under the same task design as T. This framework has become the de facto standard for assessing progress in MI research, used across nearly all recent MI studies without question. In this paper, we present the first in-depth study of this evaluation framework. In particular, we identify a critical issue of this standard framework: Type-I adversarial examples. These are reconstructions that do not capture the visual features of private training data, yet are still deemed successful by T and ultimately transferable to E. Such false positives undermine the reliability of the standard MI evaluation framework. To address this issue, we introduce a new MI evaluation framework that replaces the evaluation model E with advanced Multimodal Large Language Models (MLLMs). By leveraging their general-purpose visual understanding, our MLLM-based framework does not depend on training of shared task design as in T, thus reducing Type-I transferability and providing more faithful assessments of reconstruction success. Using our MLLM-based evaluation framework, we reevaluate 27 diverse MI attack setups and empirically reveal consistently high false positive rates under the standard evaluation framework. Importantly, we demonstrate that many state-of-the-art (SOTA) MI methods report inflated attack accuracy, indicating that actual privacy leakage is significantly lower than previously believed. By uncovering this critical issue and proposing a robust solution, our work enables a reassessment of progress in MI research and sets a new standard for reliable and robust evaluation. Code can be found in https://github.com/hosytuyen/MI-Eval-MLLM

摘要: 模型倒置（MI）攻击旨在通过利用对机器学习模型T的访问来从私人训练数据中重建信息。为了评估此类攻击，标准评估框架依赖于评估模型E，该模型在与T相同的任务设计下训练。该框架已成为评估MI研究进展的事实标准，几乎所有最近的MI研究都毫无疑问地使用了该框架。在本文中，我们对该评估框架进行了首次深入研究。特别是，我们确定了这个标准框架的一个关键问题：I型对抗性示例。这些重建并没有捕捉到私人训练数据的视觉特征，但仍然被T认为是成功的，并最终转移到E。这种假阳性损害了标准管理信息评价框架的可靠性。为了解决这个问题，我们引入了一个新的MI评估框架，用先进的多模态大型语言模型（MLLM）取代了评估模型E。通过利用他们的通用视觉理解，我们基于MLLM的框架不依赖于T中的共享任务设计的训练，从而降低了I型可转移性，并提供了更忠实的重建成功评估。使用我们基于MLLM的评估框架，我们重新评估了27种不同的MI攻击设置，并根据经验揭示了标准评估框架下一贯的高误报率。重要的是，我们证明了许多最先进的（SOTA）MI方法报告了夸大的攻击准确性，这表明实际的隐私泄露显着低于以前认为的。通过揭示这一关键问题并提出一个强有力的解决方案，我们的工作能够重新评估MI研究的进展，并为可靠和强大的评估设定了新的标准。代码可在https://github.com/hosytuyen/MI-Eval-MLLM找到



## **43. Quantifying Privacy Leakage in Split Inference via Fisher-Approximated Shannon Information Analysis**

通过费舍尔逼近的香农信息分析量化分裂推理中的隐私泄露 cs.CR

13pages, 12 figures

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2504.10016v2) [paper-pdf](https://arxiv.org/pdf/2504.10016v2)

**Authors**: Ruijun Deng, Zhihui Lu, Qiang Duan, Shijing Hu

**Abstract**: Split inference (SI) partitions deep neural networks into distributed sub-models, enabling collaborative learning without directly sharing raw data. However, SI remains vulnerable to Data Reconstruction Attacks (DRAs), where adversaries exploit exposed smashed data to recover private inputs. Despite substantial progress in attack-defense methodologies, the fundamental quantification of privacy risks is still underdeveloped. This paper establishes an information-theoretic framework for privacy leakage in SI, defining leakage as the adversary's certainty and deriving both average-case and worst-case error lower bounds. We further introduce Fisher-approximated Shannon information (FSInfo), a new privacy metric based on Fisher Information (FI) that enables operational and tractable computation of privacy leakage. Building on this metric, we develop FSInfoGuard, a defense mechanism that achieves a strong privacy-utility tradeoff. Our empirical study shows that FSInfo is an effective privacy metric across datasets, models, and defense strengths, providing accurate privacy estimates that support the design of defense methods outperforming existing approaches in both privacy protection and utility preservation. The code is available at https://github.com/SASA-cloud/FSInfo.

摘要: 分裂推理（SI）将深度神经网络划分为分布式子模型，无需直接共享原始数据即可实现协作学习。然而，SI仍然容易受到数据重建攻击（DSA）的影响，对手利用暴露的破碎数据来恢复私人输入。尽管在攻击防御方法方面取得了重大进展，但隐私风险的基本量化仍然不够充分。本文建立了SI中隐私泄露的信息理论框架，将泄露定义为对手的确定性，并推导出平均情况和最坏情况的错误下限。我们进一步引入了费舍尔逼近的香农信息（FSInfo），这是一种基于费舍尔信息（FI）的新隐私指标，可以对隐私泄露进行可操作且易于处理的计算。在此指标的基础上，我们开发了FSInfoGuard，这是一种实现强大的隐私与公用事业权衡的防御机制。我们的实证研究表明，FSInfo是跨数据集、模型和防御优势的有效隐私指标，可以提供准确的隐私估计，支持防御方法的设计在隐私保护和效用保护方面优于现有方法。该代码可在https://github.com/SASA-cloud/FSInfo上获取。



## **44. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

一张图片即可：用单个图像毒害视觉文档检索增强生成 cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2504.02132v3) [paper-pdf](https://arxiv.org/pdf/2504.02132v3)

**Authors**: Ezzeldin Shereen, Dan Ristea, Shae McFadden, Burak Hasircioglu, Vasilios Mavroudis, Chris Hicks

**Abstract**: Retrieval-augmented generation (RAG) is instrumental for inhibiting hallucinations in large language models (LLMs) through the use of a factual knowledge base (KB). Although PDF documents are prominent sources of knowledge, text-based RAG pipelines are ineffective at capturing their rich multi-modal information. In contrast, visual document RAG (VD-RAG) uses screenshots of document pages as the KB, which has been shown to achieve state-of-the-art results. However, by introducing the image modality, VD-RAG introduces new attack vectors for adversaries to disrupt the system by injecting malicious documents into the KB. In this paper, we demonstrate the vulnerability of VD-RAG to poisoning attacks targeting both retrieval and generation. We define two attack objectives and demonstrate that both can be realized by injecting only a single adversarial image into the KB. Firstly, we introduce a targeted attack against one or a group of queries with the goal of spreading targeted disinformation. Secondly, we present a universal attack that, for any potential user query, influences the response to cause a denial-of-service in the VD-RAG system. We investigate the two attack objectives under both white-box and black-box assumptions, employing a multi-objective gradient-based optimization approach as well as prompting state-of-the-art generative models. Using two visual document datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (vision language models), we show VD-RAG is vulnerable to poisoning attacks in both the targeted and universal settings, yet demonstrating robustness to black-box attacks in the universal setting.

摘要: 检索增强生成（RAG）有助于通过使用事实知识库（KB）来抑制大型语言模型（LLM）中的幻觉。尽管PDF文档是重要的知识来源，但基于文本的RAG管道在捕获其丰富的多模式信息方面效率低下。相比之下，视觉文档RAG（VD-RAG）使用文档页面的屏幕截图作为KB，已被证明可以实现最先进的结果。然而，通过引入图像模式，VD-RAG为对手引入了新的攻击载体，通过将恶意文档注入知识库来破坏系统。在本文中，我们展示了VD-RAG对针对检索和生成的中毒攻击的脆弱性。我们定义了两个攻击目标，并证明这两个目标都可以通过仅向知识库中注入单个对抗图像来实现。首先，我们对一个或一组查询引入有针对性的攻击，目标是传播有针对性的虚假信息。其次，我们提出了一种通用攻击，对于任何潜在的用户查询，该攻击都会影响响应，从而导致VD-RAG系统中的拒绝服务。我们调查的两个攻击目标下的白盒和黑盒的假设，采用多目标的基于梯度的优化方法，以及促使国家的最先进的生成模型。使用两个可视化文档数据集，一组不同的最先进的检索器（嵌入模型）和生成器（视觉语言模型），我们表明VD-RAG在目标和通用设置中都容易受到中毒攻击，但在通用设置中表现出对黑盒攻击的鲁棒性。



## **45. A Closer Look at Adversarial Suffix Learning for Jailbreaking LLMs: Augmented Adversarial Trigger Learning**

深入了解越狱LLM的对抗性后缀学习：增强对抗性触发学习 cs.LG

the Association for Computational Linguistics: NAACL 2025

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2503.12339v4) [paper-pdf](https://arxiv.org/pdf/2503.12339v4)

**Authors**: Zhe Wang, Yanjun Qi

**Abstract**: Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs. We released our code https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning

摘要: 基于梯度优化的对抗攻击方法自动学习对抗触发器，以生成越狱提示或泄露系统提示。在这项工作中，我们仔细研究了对抗性触发学习的优化目标，并提出了ATLA：具有增强目标的对抗性触发学习。ATLA将之前研究使用的负对似然损失改进为加权损失公式，该公式鼓励学习的对抗触发因素对响应格式代币进行更多优化。这使得ATLA能够仅从一个查询-响应对中学习对抗触发器，并且学习到的触发器可以很好地推广到其他类似的查询。我们进一步设计了一个变体，通过抑制规避反应的辅助损失来增强触发优化。我们展示了如何使用ATLA来学习对抗性后缀、越狱LLM并提取隐藏的系统提示。从经验上看，我们证明ATLA始终优于当前最先进的技术，在攻击方面取得了近100%的成功，同时所需的查询减少了80%。ATLA学习越狱后缀表现出对未见查询的高度概括，并很好地转移到新的LLM。我们发布了我们的代码https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning



## **46. Adversarial Agents: Black-Box Evasion Attacks with Reinforcement Learning**

对抗性代理：利用强化学习进行黑匣子规避攻击 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2503.01734v2) [paper-pdf](https://arxiv.org/pdf/2503.01734v2)

**Authors**: Kyle Domico, Jean-Charles Noirot Ferrand, Ryan Sheatsley, Eric Pauley, Josiah Hanna, Patrick McDaniel

**Abstract**: Attacks on machine learning models have been extensively studied through stateless optimization. In this paper, we demonstrate how a reinforcement learning (RL) agent can learn a new class of attack algorithms that generate adversarial samples. Unlike traditional adversarial machine learning (AML) methods that craft adversarial samples independently, our RL-based approach retains and exploits past attack experience to improve the effectiveness and efficiency of future attacks. We formulate adversarial sample generation as a Markov Decision Process and evaluate RL's ability to (a) learn effective and efficient attack strategies and (b) compete with state-of-the-art AML. On two image classification benchmarks, our agent increases attack success rate by up to 13.2% and decreases the average number of victim model queries per attack by up to 16.9% from the start to the end of training. In a head-to-head comparison with state-of-the-art image attacks, our approach enables an adversary to generate adversarial samples with 17% more success on unseen inputs post-training. From a security perspective, this work demonstrates a powerful new attack vector that uses RL to train agents that attack ML models efficiently and at scale.

摘要: 对机器学习模型的攻击已经通过无状态优化进行了广泛的研究。在本文中，我们演示了强化学习（RL）代理如何学习一类新的攻击算法，生成对抗性样本。与独立制作对抗样本的传统对抗机器学习（AML）方法不同，我们基于RL的方法保留并利用过去的攻击经验来提高未来攻击的有效性和效率。我们将对抗性样本生成公式化为马尔可夫决策过程，并评估RL的能力：（a）学习有效和高效的攻击策略;（b）与最先进的AML竞争。在两个图像分类基准上，从训练开始到结束，我们的代理将攻击成功率提高了高达13.2%，并将每次攻击的平均受害者模型查询数量减少了高达16.9%。与最先进的图像攻击进行正面比较，我们的方法使对手能够生成对抗样本，训练后在未见输入上的成功率高出17%。从安全角度来看，这项工作展示了一种强大的新攻击载体，它使用RL来训练有效、大规模攻击ML模型的代理。



## **47. 1-Lipschitz Network Initialization for Certifiably Robust Classification Applications: A Decay Problem**

1-Lipschitz网络可证明鲁棒分类应用程序：一个衰变问题 cs.LG

15 pages, 11 figures; added additional experimental results and formatted to Elsevier format

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2503.00240v2) [paper-pdf](https://arxiv.org/pdf/2503.00240v2)

**Authors**: Marius F. R. Juston, Ramavarapu S. Sreenivas, William R. Norris, Dustin Nottage, Ahmet Soylemezoglu

**Abstract**: This paper discusses the weight parametrization of two standard 1-Lipschitz network architectures, the Almost-Orthogonal-Layers (AOL) and the SDP-based Lipschitz Layers (SLL). It examines their impact on initialization for deep 1-Lipschitz feedforward networks, and discusses underlying issues surrounding this initialization. These networks are mainly used in certifiably robust classification applications to combat adversarial attacks by limiting the impact of perturbations on the classification output. Exact and upper bounds for the parameterized weight variance were calculated assuming a standard Normal distribution initialization; additionally, an upper bound was computed assuming a Generalized Normal Distribution, generalizing the proof for Uniform, Laplace, and Normal distribution weight initializations. It is demonstrated that the weight variance holds no bearing on the output variance distribution and that only the dimension of the weight matrices matters. Additionally, this paper demonstrates that the weight initialization always causes deep 1-Lipschitz networks to decay to zero.

摘要: 本文讨论了两种标准的1-Lipschitz网络结构，几乎正交层（AOL）和基于SDP的Lipschitz层（SLL）的权重参数化。它研究了它们对深度1-Lipschitz前馈网络初始化的影响，并讨论了围绕该初始化的基本问题。这些网络主要用于可证明的鲁棒分类应用，通过限制扰动对分类输出的影响来对抗性攻击。假设标准正态分布初始化，计算参数化权重方差的精确和上限;此外，假设广义正态分布，计算上限，推广均匀、拉普拉斯和正态分布权重初始化的证明。结果表明，权重方差对输出方差分布没有影响，只有权重矩阵的维数有关系。此外，本文证明了权重初始化总是导致深度1-Lipschitz网络衰减到零。



## **48. Eguard: Defending LLM Embeddings Against Inversion Attacks via Text Mutual Information Optimization**

Eguard：通过文本互信息优化保护LLM嵌入免受倒置攻击 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2411.05034v2) [paper-pdf](https://arxiv.org/pdf/2411.05034v2)

**Authors**: Tiantian Liu, Hongwei Yao, Feng Lin, Tong Wu, Zhan Qin, Kui Ren

**Abstract**: Embeddings have become a cornerstone in the functionality of large language models (LLMs) due to their ability to transform text data into rich, dense numerical representations that capture semantic and syntactic properties. These embedding vector databases serve as the long-term memory of LLMs, enabling efficient handling of a wide range of natural language processing tasks. However, the surge in popularity of embedding vector databases in LLMs has been accompanied by significant concerns about privacy leakage. Embedding vector databases are particularly vulnerable to embedding inversion attacks, where adversaries can exploit the embeddings to reverse-engineer and extract sensitive information from the original text data. Existing defense mechanisms have shown limitations, often struggling to balance security with the performance of downstream tasks. To address these challenges, we introduce Eguard, a novel defense mechanism designed to mitigate embedding inversion attacks. Eguard employs a transformer-based projection network and text mutual information optimization to safeguard embeddings while preserving the utility of LLMs. Our approach significantly reduces privacy risks, protecting over 95% of tokens from inversion while maintaining high performance across downstream tasks consistent with original embeddings.

摘要: 嵌入已成为大型语言模型（LLM）功能的基石，因为它们能够将文本数据转换为捕获语义和语法属性的丰富、密集的数字表示。这些嵌入式载体数据库充当LLM的长期存储器，能够高效处理广泛的自然语言处理任务。然而，在LLM中嵌入载体数据库的普及率激增，同时也伴随着对隐私泄露的严重担忧。嵌入式载体数据库特别容易受到嵌入倒置攻击，对手可以利用嵌入进行反向工程并从原始文本数据中提取敏感信息。现有的防御机制已经显示出局限性，经常难以平衡安全性与下游任务的性能。为了解决这些挑战，我们引入了Eguard，这是一种新颖的防御机制，旨在减轻嵌入倒置攻击。Eguard采用基于转换器的投影网络和文本互信息优化来保护嵌入，同时保留LLM的实用性。我们的方法显着降低了隐私风险，保护超过95%的令牌免受倒置，同时在下游任务中保持与原始嵌入一致的高性能。



## **49. Sparse-PGD: A Unified Framework for Sparse Adversarial Perturbations Generation**

Sparse-PGD：稀疏对抗扰动生成的统一框架 cs.LG

Accepted by TPAMI

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2405.05075v4) [paper-pdf](https://arxiv.org/pdf/2405.05075v4)

**Authors**: Xuyang Zhong, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations, including both unstructured and structured ones. We propose a framework based on a white-box PGD-like attack method named Sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine Sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against unstructured and structured sparse adversarial perturbations. Moreover, the efficiency of Sparse-PGD enables us to conduct adversarial training to build robust models against various sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks. Codes are available at https://github.com/CityU-MLO/sPGD.

摘要: 这项工作研究了稀疏的对抗性扰动，包括非结构化和结构化的扰动。我们提出了一个基于类似白盒PGD攻击方法的框架，名为Sparse-PVD，以有效且高效地生成此类扰动。此外，我们将Sparse-PGDD与黑匣子攻击相结合，以全面、更可靠地评估模型对非结构化和结构化稀疏对抗扰动的鲁棒性。此外，Sparse-PVD的效率使我们能够进行对抗训练，以针对各种稀疏扰动构建稳健的模型。大量实验表明，我们提出的攻击算法在不同场景下表现出强大的性能。更重要的是，与其他稳健模型相比，我们的对抗训练模型表现出了针对各种稀疏攻击的最新稳健性。代码可访问https://github.com/CityU-MLO/sPGD。



## **50. High Dimensional Distributed Gradient Descent with Arbitrary Number of Byzantine Attackers**

任意数量的拜占庭攻击者的多维分布式梯度下降 cs.LG

25 pages, 4 figures

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2307.13352v3) [paper-pdf](https://arxiv.org/pdf/2307.13352v3)

**Authors**: Wenyu Liu, Tianqiang Huang, Pengfei Zhang, Zong Ke, Minghui Min, Puning Zhao

**Abstract**: Adversarial attacks pose a major challenge to distributed learning systems, prompting the development of numerous robust learning methods. However, most existing approaches suffer from the curse of dimensionality, i.e. the error increases with the number of model parameters. In this paper, we make a progress towards high dimensional problems, under arbitrary number of Byzantine attackers. The cornerstone of our design is a direct high dimensional semi-verified mean estimation method. The idea is to identify a subspace with large variance. The components of the mean value perpendicular to this subspace are estimated using corrupted gradient vectors uploaded from worker machines, while the components within this subspace are estimated using auxiliary dataset. As a result, a combination of large corrupted dataset and small clean dataset yields significantly better performance than using them separately. We then apply this method as the aggregator for distributed learning problems. The theoretical analysis shows that compared with existing solutions, our method gets rid of $\sqrt{d}$ dependence on the dimensionality, and achieves minimax optimal statistical rates. Numerical results validate our theory as well as the effectiveness of the proposed method.

摘要: 对抗性攻击对分布式学习系统构成了重大挑战，促使许多稳健的学习方法的开发。然而，大多数现有方法都遭受维度诅咒，即误差随着模型参数的数量而增加。在本文中，我们在任意数量的拜占庭攻击者的情况下向多维问题取得了进展。我们设计的基石是一种直接的多维半验证均值估计方法。其想法是识别具有大方差的子空间。垂直于该子空间的平均值分量是使用从工人机器上传的损坏的梯度载体来估计的，而该子空间内的分量是使用辅助数据集来估计的。因此，大型损坏数据集和小型干净数据集的组合比单独使用它们会产生明显更好的性能。然后我们将这种方法应用于分布式学习问题的聚合器。理论分析表明，与现有解决方案相比，我们的方法摆脱了$\SQRT{d}$对维度的依赖，实现了极小极大最优统计率。数值结果验证了我们的理论以及所提出方法的有效性。




# Latest Adversarial Attack Papers
**update at 2025-04-09 10:27:08**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Exploring Adversarial Obstacle Attacks in Search-based Path Planning for Autonomous Mobile Robots**

探索自主移动机器人基于搜索的路径规划中的对抗障碍攻击 cs.RO

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.06154v1) [paper-pdf](http://arxiv.org/pdf/2504.06154v1)

**Authors**: Adrian Szvoren, Jianwei Liu, Dimitrios Kanoulas, Nilufer Tuptuk

**Abstract**: Path planning algorithms, such as the search-based A*, are a critical component of autonomous mobile robotics, enabling robots to navigate from a starting point to a destination efficiently and safely. We investigated the resilience of the A* algorithm in the face of potential adversarial interventions known as obstacle attacks. The adversary's goal is to delay the robot's timely arrival at its destination by introducing obstacles along its original path.   We developed malicious software to execute the attacks and conducted experiments to assess their impact, both in simulation using TurtleBot in Gazebo and in real-world deployment with the Unitree Go1 robot. In simulation, the attacks resulted in an average delay of 36\%, with the most significant delays occurring in scenarios where the robot was forced to take substantially longer alternative paths. In real-world experiments, the delays were even more pronounced, with all attacks successfully rerouting the robot and causing measurable disruptions. These results highlight that the algorithm's robustness is not solely an attribute of its design but is significantly influenced by the operational environment. For example, in constrained environments like tunnels, the delays were maximized due to the limited availability of alternative routes.

摘要: 路径规划算法（例如基于搜索的A*）是自主移动机器人技术的关键组成部分，使机器人能够高效、安全地从起点导航到目的地。我们研究了A* 算法在面临潜在的对抗性干预（即障碍攻击）时的弹性。对手的目标是通过在机器人的原始路径上设置障碍物来推迟机器人及时到达目的地。   我们开发了恶意软件来执行攻击，并进行了实验来评估其影响，无论是在Gazebo中使用TurtleBot进行模拟还是在现实世界中使用Unitree Go 1机器人进行部署。在模拟中，攻击导致平均延迟为36%，其中最显着的延迟发生在机器人被迫采取更长的替代路径的情况下。在现实世界的实验中，延迟甚至更加明显，所有攻击都成功地改变了机器人的路线并造成了可测量的中断。这些结果凸显了该算法的鲁棒性不仅仅是其设计的属性，而且还受到操作环境的显着影响。例如，在隧道等受限环境中，由于替代路线的可用性有限，延误被最大化。



## **2. Frequency maps reveal the correlation between Adversarial Attacks and Implicit Bias**

频率图揭示了对抗性攻击和隐性偏见之间的相关性 cs.LG

Accepted at IJCNN 2025

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2305.15203v3) [paper-pdf](http://arxiv.org/pdf/2305.15203v3)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto d'Onofrio, Luca Manzoni, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification tasks, neural networks are known to be vulnerable to adversarial attacks, subtle perturbations of the input data designed to deceive the model. In this work, we investigate the correlation between these perturbations and the implicit bias of neural networks trained with gradient-based algorithms. To this end, we analyse a representation of the network's implicit bias through the lens of the Fourier transform. Specifically, we identify unique fingerprints of implicit bias and adversarial attacks by calculating the minimal, essential frequencies needed for accurate classification of each image, as well as the frequencies that drive misclassification in its adversarially perturbed counterpart. This approach enables us to uncover and analyse the correlation between these essential frequencies, providing a precise map of how the network's biases align or contrast with the frequency components exploited by adversarial attacks. To this end, among other methods, we use a newly introduced technique capable of detecting nonlinear correlations between high-dimensional datasets. Our results provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are highly correlated and suggest new potential strategies for adversarial defence.

摘要: 尽管神经网络在分类任务中的表现令人印象深刻，但众所周知，神经网络很容易受到对抗攻击，即旨在欺骗模型的输入数据的微妙扰动。在这项工作中，我们研究了这些扰动与用基于梯度的算法训练的神经网络的隐式偏差之间的相关性。为此，我们通过傅里叶变换的镜头分析网络隐含偏差的表示。具体来说，我们通过计算对每张图像进行准确分类所需的最小、基本频率，以及导致其受对抗干扰的对应图像中误分类的频率，来识别隐性偏见和对抗攻击的独特指纹。这种方法使我们能够发现和分析这些基本频率之间的相关性，提供网络偏差如何与对抗性攻击利用的频率分量对齐或对比的精确地图。为此，除其他方法外，我们使用了一种新引入的能够检测多维数据集之间非线性相关性的技术。我们的结果提供了经验证据，证明傅里叶空间中的网络偏差和对抗性攻击的目标频率高度相关，并提出了对抗性防御的新潜在策略。



## **3. Mind the Trojan Horse: Image Prompt Adapter Enabling Scalable and Deceptive Jailbreaking**

小心特洛伊木马：图像提示适配器支持可扩展和欺骗性越狱 cs.CV

Accepted by CVPR2025 as Highlight

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05838v1) [paper-pdf](http://arxiv.org/pdf/2504.05838v1)

**Authors**: Junxi Chen, Junhao Dong, Xiaohua Xie

**Abstract**: Recently, the Image Prompt Adapter (IP-Adapter) has been increasingly integrated into text-to-image diffusion models (T2I-DMs) to improve controllability. However, in this paper, we reveal that T2I-DMs equipped with the IP-Adapter (T2I-IP-DMs) enable a new jailbreak attack named the hijacking attack. We demonstrate that, by uploading imperceptible image-space adversarial examples (AEs), the adversary can hijack massive benign users to jailbreak an Image Generation Service (IGS) driven by T2I-IP-DMs and mislead the public to discredit the service provider. Worse still, the IP-Adapter's dependency on open-source image encoders reduces the knowledge required to craft AEs. Extensive experiments verify the technical feasibility of the hijacking attack. In light of the revealed threat, we investigate several existing defenses and explore combining the IP-Adapter with adversarially trained models to overcome existing defenses' limitations. Our code is available at https://github.com/fhdnskfbeuv/attackIPA.

摘要: 最近，图像提示适配器（IP适配器）越来越多地集成到文本到图像扩散模型（T2 I-DM）中，以提高可控性。然而，在本文中，我们揭示了配备IP适配器（T2 I-IP-DMs）的T2 I-DM会启用一种名为劫持攻击的新越狱攻击。我们证明，通过上传难以察觉的图像空间对抗示例（AE），对手可以劫持大量良性用户来越狱由T2 I-IP-DM驱动的图像生成服务（IRS），并误导公众抹黑服务提供商。更糟糕的是，IP适配器对开源图像编码器的依赖减少了制作AE所需的知识。大量实验验证了劫持攻击的技术可行性。鉴于所揭示的威胁，我们调查了几种现有的防御措施，并探索将IP适配器与对抗训练模型相结合，以克服现有防御措施的局限性。我们的代码可在https://github.com/fhdnskfbeuv/attackIPA上获取。



## **4. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

StealthRank：通过StealthPropriation优化进行LLM排名操纵 cs.IR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05804v1) [paper-pdf](http://arxiv.org/pdf/2504.05804v1)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems.

摘要: 将大型语言模型（LLM）集成到信息检索系统中引入了新的攻击表面，特别是对于对抗性排名操纵。我们介绍了StealthRank，这是一种新型的对抗性排名攻击，它可以操纵LLM驱动的产品推荐系统，同时保持文本流畅性和隐蔽性。与经常引入可检测异常的现有方法不同，StealthRank采用基于能量的优化框架与Langevin动态相结合来生成StealthRank脚本（SPP）-嵌入产品描述中的对抗性文本序列，微妙而有效地影响LLM排名机制。我们在多个LLM中评估StealthRank，证明其能够秘密提高目标产品的排名，同时避免容易检测到的显式操纵痕迹。我们的结果表明，StealthRank在有效性和隐蔽性方面始终优于最先进的对抗排名基线，凸显了LLM驱动的推荐系统中的关键漏洞。



## **5. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

用于机器学习文本分类器的自动可信度Oracle生成 cs.SE

Accepted to FSE 2025

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2410.22663v2) [paper-pdf](http://arxiv.org/pdf/2410.22663v2)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Studies indicate that conventional metrics are insufficient to build human trust in ML models. These models often learn spurious correlations and predict based on them. In the real world, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable based on valid patterns in the data. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods. However, this is time-consuming, error-prone, and unscalable.   We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers. TOKI automatically checks whether the words contributing the most to a prediction are semantically related to the predicted class. Specifically, we leverage ML explanations to extract the decision-contributing words and measure their semantic relatedness with the class based on word embeddings. We also introduce a novel adversarial attack method that targets trustworthiness vulnerabilities identified by TOKI. To evaluate their alignment with human judgement, experiments are conducted. We compare TOKI with a naive baseline based solely on model confidence and TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot effectively distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided attack method is more effective with fewer perturbations than A2T.

摘要: 用于文本分类的机器学习（ML）已广泛应用于各个领域。这些应用程序可能会显着影响道德、经济和人类行为，从而引发人们对信任ML决策的严重担忧。研究表明，传统指标不足以建立人类对ML模型的信任。这些模型经常学习虚假相关性并基于它们进行预测。在现实世界中，他们的表现可能会显着恶化。为了避免这种情况，常见的做法是根据数据中的有效模式测试预测是否合理。与此同时，还引入了一个称为可信度Oracle问题的挑战。由于缺乏自动可信度预言，评估需要对解释方法披露的决策过程进行手动验证。然而，这耗时、容易出错且不可扩展。   我们提出了TOKI，这是第一种文本分类器的自动可信Oracle生成方法。TOKI自动检查对预测贡献最大的单词是否与预测的类别在语义上相关。具体来说，我们利用ML解释来提取影响决策的单词，并基于单词嵌入来测量它们与类的语义相关性。我们还引入了一种新颖的对抗攻击方法，该方法针对TOKI识别的可信度漏洞。为了评估它们与人类判断的一致性，我们进行了实验。我们将TOKI与仅基于模型置信度和TOKI引导的对抗攻击方法与A2 T（一种SOTA对抗攻击方法）进行比较。结果表明，依赖预测不确定性无法有效区分可信和不可信的预测，TOKI的准确性比原始基线高出142%，TOKI引导的攻击方法比A2 T更有效，干扰更少。



## **6. Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing**

Nes 2Net：基础模型驱动语音反欺骗的轻量级嵌套架构 eess.AS

This manuscript has been submitted for peer review

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05657v1) [paper-pdf](http://arxiv.org/pdf/2504.05657v1)

**Authors**: Tianchi Liu, Duc-Tuan Truong, Rohan Kumar Das, Kong Aik Lee, Haizhou Li

**Abstract**: Speech foundation models have significantly advanced various speech-related tasks by providing exceptional representation capabilities. However, their high-dimensional output features often create a mismatch with downstream task models, which typically require lower-dimensional inputs. A common solution is to apply a dimensionality reduction (DR) layer, but this approach increases parameter overhead, computational costs, and risks losing valuable information. To address these issues, we propose Nested Res2Net (Nes2Net), a lightweight back-end architecture designed to directly process high-dimensional features without DR layers. The nested structure enhances multi-scale feature extraction, improves feature interaction, and preserves high-dimensional information. We first validate Nes2Net on CtrSVDD, a singing voice deepfake detection dataset, and report a 22% performance improvement and an 87% back-end computational cost reduction over the state-of-the-art baseline. Additionally, extensive testing across four diverse datasets: ASVspoof 2021, ASVspoof 5, PartialSpoof, and In-the-Wild, covering fully spoofed speech, adversarial attacks, partial spoofing, and real-world scenarios, consistently highlights Nes2Net's superior robustness and generalization capabilities. The code package and pre-trained models are available at https://github.com/Liu-Tianchi/Nes2Net.

摘要: 语音基础模型通过提供出色的表示能力，显着推进了各种语音相关任务。然而，它们的多维输出特征通常会与下游任务模型产生不匹配，下游任务模型通常需要较低维度的输入。常见的解决方案是应用降维（DR）层，但这种方法增加了参数负担、计算成本，并存在丢失有价值信息的风险。为了解决这些问题，我们提出了Nested Res 2Net（Nes 2Net），这是一种轻量级的后台架构，旨在直接处理多维特征，而无需DR层。嵌套结构增强了多尺度特征提取，改善了特征交互，并保留了多维信息。我们首先在CtrSVD（歌唱声深度伪造检测数据集）上验证了Nes 2Net，并报告与最先进的基线相比，性能提高了22%，后台计算成本降低了87%。此外，对四个不同数据集进行了广泛的测试：ASVspoof 2021、ASVspoof 5、PartialSpoof和In-the-Wild，涵盖了完全欺骗的语音、对抗性攻击、部分欺骗和现实世界场景，一致强调了Nes 2Net卓越的鲁棒性和概括能力。代码包和预训练模型可在https://github.com/Liu-Tianchi/Nes2Net上获取。



## **7. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

糖衣毒药：良性一代解锁法学硕士越狱 cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05652v1) [paper-pdf](http://arxiv.org/pdf/2504.05652v1)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Jie-Zhang

**Abstract**: Large Language Models (LLMs) have become increasingly integral to a wide range of applications. However, they still remain the threat of jailbreak attacks, where attackers manipulate designed prompts to make the models elicit malicious outputs. Analyzing jailbreak methods can help us delve into the weakness of LLMs and improve it. In this paper, We reveal a vulnerability in large language models (LLMs), which we term Defense Threshold Decay (DTD), by analyzing the attention weights of the model's output on input and subsequent output on prior output: as the model generates substantial benign content, its attention weights shift from the input to prior output, making it more susceptible to jailbreak attacks. To demonstrate the exploitability of DTD, we propose a novel jailbreak attack method, Sugar-Coated Poison (SCP), which induces the model to generate substantial benign content through benign input and adversarial reasoning, subsequently producing malicious content. To mitigate such attacks, we introduce a simple yet effective defense strategy, POSD, which significantly reduces jailbreak success rates while preserving the model's generalization capabilities.

摘要: 大型语言模型（LLM）已经成为越来越广泛的应用程序的组成部分。然而，它们仍然是越狱攻击的威胁，攻击者操纵设计的提示，使模型引发恶意输出。分析越狱方法可以帮助我们深入研究LLM的弱点并对其进行改进。本文通过分析模型的输出对输入和后续输出对先前输出的注意力权重，揭示了大型语言模型（LLM）中的一个漏洞，我们称之为防御阈值衰减（DTD）：当模型生成大量良性内容时，其注意力权重从输入转移到先前输出，使其更容易受到越狱攻击。为了证明DTD的可利用性，我们提出了一种新的越狱攻击方法，糖衣毒药（SCP），它诱导模型通过良性输入和对抗性推理生成大量良性内容，随后产生恶意内容。为了减轻这种攻击，我们引入了一种简单而有效的防御策略POSD，它可以显着降低越狱成功率，同时保留模型的泛化能力。



## **8. SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments**

SceneRAP：针对现实世界环境中视觉语言模型的场景一致印刷对抗规划器 cs.CV

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2412.00114v2) [paper-pdf](http://arxiv.org/pdf/2412.00114v2)

**Authors**: Yue Cao, Yun Xing, Jie Zhang, Di Lin, Tianwei Zhang, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in interpreting visual content. While existing works demonstrate these models' vulnerability to deliberately placed adversarial texts, such texts are often easily identifiable as anomalous. In this paper, we present the first approach to generate scene-coherent typographic adversarial attacks that mislead advanced LVLMs while maintaining visual naturalness through the capability of the LLM-based agent. Our approach addresses three critical questions: what adversarial text to generate, where to place it within the scene, and how to integrate it seamlessly. We propose a training-free, multi-modal LLM-driven scene-coherent typographic adversarial planning (SceneTAP) that employs a three-stage process: scene understanding, adversarial planning, and seamless integration. The SceneTAP utilizes chain-of-thought reasoning to comprehend the scene, formulate effective adversarial text, strategically plan its placement, and provide detailed instructions for natural integration within the image. This is followed by a scene-coherent TextDiffuser that executes the attack using a local diffusion mechanism. We extend our method to real-world scenarios by printing and placing generated patches in physical environments, demonstrating its practical implications. Extensive experiments show that our scene-coherent adversarial text successfully misleads state-of-the-art LVLMs, including ChatGPT-4o, even after capturing new images of physical setups. Our evaluations demonstrate a significant increase in attack success rates while maintaining visual naturalness and contextual appropriateness. This work highlights vulnerabilities in current vision-language models to sophisticated, scene-coherent adversarial attacks and provides insights into potential defense mechanisms.

摘要: 大型视觉语言模型（LVLM）在解释视觉内容方面表现出了非凡的能力。虽然现有的作品证明了这些模型对故意放置的对抗文本的脆弱性，但此类文本通常很容易被识别为异常文本。在本文中，我们提出了第一种生成场景一致印刷对抗攻击的方法，这种攻击可以误导高级LVLM，同时通过基于LLM的代理的能力保持视觉自然性。我们的方法解决了三个关键问题：生成什么对抗文本、将其放置在场景中的位置以及如何无缝集成它。我们提出了一种免培训、多模式LLM驱动的场景一致印刷对抗性规划（SceneRAP），该规划采用三阶段流程：场景理解、对抗性规划和无缝集成。SceneRAP利用思想链推理来理解场景、制定有效的对抗文本、战略性地规划其放置，并为图像中的自然整合提供详细的说明。随后是场景一致的文本扩散用户，它使用本地扩散机制执行攻击。我们通过打印并将生成的补丁放置在物理环境中，将我们的方法扩展到现实世界场景，展示其实际含义。大量实验表明，即使在捕获物理设置的新图像之后，我们的场景连贯对抗文本也能成功误导最先进的LVLM，包括ChatGPT-4 o。我们的评估表明，攻击成功率显着提高，同时保持视觉自然性和上下文适当性。这项工作强调了当前视觉语言模型对复杂、场景一致的对抗攻击的脆弱性，并提供了对潜在防御机制的见解。



## **9. ShadowCoT: Cognitive Hijacking for Stealthy Reasoning Backdoors in LLMs**

ShadowCoT：LLM中秘密推理后门的认知劫持 cs.CR

Zhao et al., 16 pages, 2025, uploaded by Hanzhou Wu, Shanghai  University

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05605v1) [paper-pdf](http://arxiv.org/pdf/2504.05605v1)

**Authors**: Gejian Zhao, Hanzhou Wu, Xinpeng Zhang, Athanasios V. Vasilakos

**Abstract**: Chain-of-Thought (CoT) enhances an LLM's ability to perform complex reasoning tasks, but it also introduces new security issues. In this work, we present ShadowCoT, a novel backdoor attack framework that targets the internal reasoning mechanism of LLMs. Unlike prior token-level or prompt-based attacks, ShadowCoT directly manipulates the model's cognitive reasoning path, enabling it to hijack multi-step reasoning chains and produce logically coherent but adversarial outcomes. By conditioning on internal reasoning states, ShadowCoT learns to recognize and selectively disrupt key reasoning steps, effectively mounting a self-reflective cognitive attack within the target model. Our approach introduces a lightweight yet effective multi-stage injection pipeline, which selectively rewires attention pathways and perturbs intermediate representations with minimal parameter overhead (only 0.15% updated). ShadowCoT further leverages reinforcement learning and reasoning chain pollution (RCP) to autonomously synthesize stealthy adversarial CoTs that remain undetectable to advanced defenses. Extensive experiments across diverse reasoning benchmarks and LLMs show that ShadowCoT consistently achieves high Attack Success Rate (94.4%) and Hijacking Success Rate (88.4%) while preserving benign performance. These results reveal an emergent class of cognition-level threats and highlight the urgent need for defenses beyond shallow surface-level consistency.

摘要: 思想链（CoT）增强了LLM执行复杂推理任务的能力，但也引入了新的安全问题。在这项工作中，我们提出了ShadowCoT，这是一种针对LLM内部推理机制的新型后门攻击框架。与之前的代币级或基于预算的攻击不同，ShadowCoT直接操纵模型的认知推理路径，使其能够劫持多步推理链并产生逻辑上连贯但具有对抗性的结果。通过以内部推理状态为条件，ShadowCoT学会识别并选择性地破坏关键推理步骤，有效地在目标模型内发起自我反思认知攻击。我们的方法引入了一种轻量级但有效的多阶段注入管道，它选择性地重新连接注意力路径并以最小的参数负担（仅更新0.15%）扰乱中间表示。ShadowCoT进一步利用强化学习和推理链污染（PGP）来自主合成先进防御系统无法检测到的隐形对抗CoT。跨各种推理基准和LLM的广泛实验表明，ShadowCoT始终实现高攻击成功率（94.4%）和劫持成功率（88.4%），同时保持良性性能。这些结果揭示了一类新出现的认知层面威胁，并凸显了对超越浅层表面一致性的防御的迫切需求。



## **10. Impact Assessment of Cyberattacks in Inverter-Based Microgrids**

基于逆变器的微电网网络攻击的影响评估 eess.SY

IEEE Workshop on the Electronic Grid (eGrid 2025)

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05592v1) [paper-pdf](http://arxiv.org/pdf/2504.05592v1)

**Authors**: Kerd Topallaj, Colin McKerrell, Suraj Ramanathan, Ioannis Zografopoulos

**Abstract**: In recent years, the evolution of modern power grids has been driven by the growing integration of remotely controlled grid assets. Although Distributed Energy Resources (DERs) and Inverter-Based Resources (IBR) enhance operational efficiency, they also introduce cybersecurity risks. The remote accessibility of such critical grid components creates entry points for attacks that adversaries could exploit, posing threats to the stability of the system. To evaluate the resilience of energy systems under such threats, this study employs real-time simulation and a modified version of the IEEE 39-bus system that incorporates a Microgrid (MG) with solar-based IBR. The study assesses the impact of remote attacks impacting the MG stability under different levels of IBR penetrations through Hardware-in-the-Loop (HIL) simulations. Namely, we analyze voltage, current, and frequency profiles before, during, and after cyberattack-induced disruptions. The results demonstrate that real-time HIL testing is a practical approach to uncover potential risks and develop robust mitigation strategies for resilient MG operations.

摘要: 近年来，远程控制电网资产的日益整合推动了现代电网的发展。尽管分布式能源资源（BER）和基于逆变器的资源（IBR）提高了运营效率，但它们也带来了网络安全风险。此类关键网格组件的远程访问性为对手可能利用的攻击创建了切入点，从而对系统的稳定性构成威胁。为了评估能源系统在此类威胁下的弹性，本研究采用了实时模拟和IEEE 39节点系统的修改版本，该系统结合了微电网（MG）和基于太阳能的IBR。该研究通过硬件在环（HIL）模拟评估了远程攻击在不同IBR渗透水平下对MG稳定性的影响。也就是说，我们分析网络攻击引发的中断之前、期间和之后的电压、电流和频率分布。结果表明，实时HIL测试是发现潜在风险并为有弹性的MG运营制定稳健的缓解策略的实用方法。



## **11. Secure Diagnostics: Adversarial Robustness Meets Clinical Interpretability**

安全诊断：对抗稳健性满足临床可解释性 cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05483v1) [paper-pdf](http://arxiv.org/pdf/2504.05483v1)

**Authors**: Mohammad Hossein Najafi, Mohammad Morsali, Mohammadreza Pashanejad, Saman Soleimani Roudi, Mohammad Norouzi, Saeed Bagheri Shouraki

**Abstract**: Deep neural networks for medical image classification often fail to generalize consistently in clinical practice due to violations of the i.i.d. assumption and opaque decision-making. This paper examines interpretability in deep neural networks fine-tuned for fracture detection by evaluating model performance against adversarial attack and comparing interpretability methods to fracture regions annotated by an orthopedic surgeon. Our findings prove that robust models yield explanations more aligned with clinically meaningful areas, indicating that robustness encourages anatomically relevant feature prioritization. We emphasize the value of interpretability for facilitating human-AI collaboration, in which models serve as assistants under a human-in-the-loop paradigm: clinically plausible explanations foster trust, enable error correction, and discourage reliance on AI for high-stakes decisions. This paper investigates robustness and interpretability as complementary benchmarks for bridging the gap between benchmark performance and safe, actionable clinical deployment.

摘要: 由于违反i.i.d，用于医学图像分类的深度神经网络在临床实践中往往无法一致地概括。假设和不透明的决策。本文通过评估模型针对对抗性攻击的性能并将可解释性方法与骨科医生注释的骨折区域进行比较，研究了深度神经网络的可解释性，该网络针对骨折检测进行了微调。我们的研究结果证明，稳健的模型可以产生与临床有意义的区域更加一致的解释，这表明稳健性鼓励解剖学相关的特征优先级。我们强调可解释性对于促进人类与人工智能合作的价值，在这种合作中，模型在人在回路范式下充当助手：临床上合理的解释可以促进信任，实现纠错，并阻止对人工智能的依赖。本文研究了作为补充基准的鲁棒性和可解释性，以弥合基准性能与安全、可操作的临床部署之间的差距。



## **12. Adversarial KA**

对手KA cs.LG

8 pages, 3 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05255v1) [paper-pdf](http://arxiv.org/pdf/2504.05255v1)

**Authors**: Sviatoslav Dzhenzher, Michael H. Freedman

**Abstract**: Regarding the representation theorem of Kolmogorov and Arnold (KA) as an algorithm for representing or {\guillemotleft}expressing{\guillemotright} functions, we test its robustness by analyzing its ability to withstand adversarial attacks. We find KA to be robust to countable collections of continuous adversaries, but unearth a question about the equi-continuity of the outer functions that, so far, obstructs taking limits and defeating continuous groups of adversaries. This question on the regularity of the outer functions is relevant to the debate over the applicability of KA to the general theory of NNs.

摘要: 将Kolmogorov和Arnold（KA）的表示定理视为表示或{\guillemotleft}表达{\guillemotleft}函数的算法，我们通过分析其抵御对抗攻击的能力来测试其稳健性。我们发现KA对于连续对手的可计数集合来说是稳健的，但我们发现了一个关于外部函数的等连续性的问题，到目前为止，该问题阻碍了采取限制和击败连续对手组。这个关于外部函数规律性的问题与KA对NN一般理论的适用性的争论有关。



## **13. Security Risks in Vision-Based Beam Prediction: From Spatial Proxy Attacks to Feature Refinement**

基于视觉的射束预测中的安全风险：从空间代理攻击到特征细化 cs.NI

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05222v1) [paper-pdf](http://arxiv.org/pdf/2504.05222v1)

**Authors**: Avi Deb Raha, Kitae Kim, Mrityunjoy Gain, Apurba Adhikary, Zhu Han, Eui-Nam Huh, Choong Seon Hong

**Abstract**: The rapid evolution towards the sixth-generation (6G) networks demands advanced beamforming techniques to address challenges in dynamic, high-mobility scenarios, such as vehicular communications. Vision-based beam prediction utilizing RGB camera images emerges as a promising solution for accurate and responsive beam selection. However, reliance on visual data introduces unique vulnerabilities, particularly susceptibility to adversarial attacks, thus potentially compromising beam accuracy and overall network reliability. In this paper, we conduct the first systematic exploration of adversarial threats specifically targeting vision-based mmWave beam selection systems. Traditional white-box attacks are impractical in this context because ground-truth beam indices are inaccessible and spatial dynamics are complex. To address this, we propose a novel black-box adversarial attack strategy, termed Spatial Proxy Attack (SPA), which leverages spatial correlations between user positions and beam indices to craft effective perturbations without requiring access to model parameters or labels. To counteract these adversarial vulnerabilities, we formulate an optimization framework aimed at simultaneously enhancing beam selection accuracy under clean conditions and robustness against adversarial perturbations. We introduce a hybrid deep learning architecture integrated with a dedicated Feature Refinement Module (FRM), designed to systematically filter irrelevant, noisy and adversarially perturbed visual features. Evaluations using standard backbone models such as ResNet-50 and MobileNetV2 demonstrate that our proposed method significantly improves performance, achieving up to an +21.07\% gain in Top-K accuracy under clean conditions and a 41.31\% increase in Top-1 adversarial robustness compared to different baseline models.

摘要: 向第六代（6 G）网络的快速发展需要先进的束成形技术来应对动态、高移动性场景（例如车辆通信）中的挑战。利用Ruby相机图像的基于视觉的射束预测成为准确且响应灵敏的射束选择的有前途的解决方案。然而，对视觉数据的依赖会带来独特的漏洞，特别是对对抗攻击的敏感性，从而可能会损害射束准确性和整体网络可靠性。在本文中，我们对专门针对基于视觉的毫米波射束选择系统的对抗威胁进行了首次系统性探索。传统的白盒攻击在这种情况下是不切实际的，因为地面实况波束索引是不可访问的，空间动态是复杂的。为了解决这个问题，我们提出了一种新的黑盒对抗攻击策略，称为空间代理攻击（SPA），它利用用户位置和波束索引之间的空间相关性来制作有效的扰动，而无需访问模型参数或标签。为了抵消这些对抗性漏洞，我们制定了一个优化框架，旨在同时提高在清洁条件下的波束选择精度和对抗性扰动的鲁棒性。我们引入了一种混合深度学习架构，该架构集成了专用的特征细化模块（FRM），旨在系统地过滤不相关的、嘈杂的和不利干扰的视觉特征。使用ResNet-50和ALENetV 2等标准主干模型进行的评估表明，与不同的基线模型相比，我们提出的方法显着提高了性能，在清洁条件下Top-K准确性提高了+21.07%%，Top-1对抗鲁棒性提高了41. 31%%'。



## **14. DiffPatch: Generating Customizable Adversarial Patches using Diffusion Models**

DiffPatch：使用扩散模型生成可定制的对抗补丁 cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2412.01440v3) [paper-pdf](http://arxiv.org/pdf/2412.01440v3)

**Authors**: Zhixiang Wang, Xiaosen Wang, Bo Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can enable individuals to evade person detectors, but most existing methods prioritize attack effectiveness over stealthiness, resulting in aesthetically unpleasing patches. While generative adversarial networks and diffusion models can produce more natural-looking patches, they often fail to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these limitations, we propose DiffPatch, a novel diffusion-based framework for generating customizable and naturalistic adversarial patches. Our approach allows users to start from a reference image (rather than random noise) and incorporates masks to create patches of various shapes, not limited to squares. To preserve the original semantics during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Our method achieves attack performance comparable to state-of-the-art non-naturalistic patches while maintaining a natural appearance. Using DiffPatch, we construct AdvT-shirt-1K, the first physical adversarial T-shirt dataset comprising over a thousand images captured in diverse scenarios. AdvT-shirt-1K can serve as a useful dataset for training or testing future defense methods.

摘要: 印在衣服上的物理对抗性补丁可以使个人逃避人员检测器，但大多数现有的方法优先考虑攻击有效性而不是隐蔽性，导致美观的补丁。虽然生成式对抗网络和扩散模型可以生成更自然的补丁，但它们往往无法平衡隐蔽性和攻击有效性，并且缺乏用户定制的灵活性。为了解决这些限制，我们提出了DiffPatch，一种新的基于扩散的框架，用于生成可定制的和自然的对抗补丁。我们的方法允许用户从参考图像（而不是随机噪声）开始，并结合掩模来创建各种形状的补丁，而不限于正方形。为了在扩散过程中保留原始语义，我们采用零文本倒置将随机噪音样本映射到单个输入图像，并通过不完全扩散优化（IDO）生成补丁。我们的方法在保持自然外观的同时实现了与最先进的非自然主义补丁相当的攻击性能。使用迪夫补丁，我们构建了AdvT-shirt-1 K，这是第一个物理对抗性T恤数据集，包含在不同场景中捕获的一千多张图像。AdvT-shirt-1 K可以作为训练或测试未来防御方法的有用数据集。



## **15. Adversarial Robustness for Deep Learning-based Wildfire Prediction Models**

基于深度学习的野火预测模型的对抗鲁棒性 cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2412.20006v3) [paper-pdf](http://arxiv.org/pdf/2412.20006v3)

**Authors**: Ryo Ide, Lei Yang

**Abstract**: Rapidly growing wildfires have recently devastated societal assets, exposing a critical need for early warning systems to expedite relief efforts. Smoke detection using camera-based Deep Neural Networks (DNNs) offers a promising solution for wildfire prediction. However, the rarity of smoke across time and space limits training data, raising model overfitting and bias concerns. Current DNNs, primarily Convolutional Neural Networks (CNNs) and transformers, complicate robustness evaluation due to architectural differences. To address these challenges, we introduce WARP (Wildfire Adversarial Robustness Procedure), the first model-agnostic framework for evaluating wildfire detection models' adversarial robustness. WARP addresses inherent limitations in data diversity by generating adversarial examples through image-global and -local perturbations. Global and local attacks superimpose Gaussian noise and PNG patches onto image inputs, respectively; this suits both CNNs and transformers while generating realistic adversarial scenarios. Using WARP, we assessed real-time CNNs and Transformers, uncovering key vulnerabilities. At times, transformers exhibited over 70% precision degradation under global attacks, while both models generally struggled to differentiate cloud-like PNG patches from real smoke during local attacks. To enhance model robustness, we proposed four wildfire-oriented data augmentation techniques based on WARP's methodology and results, which diversify smoke image data and improve model precision and robustness. These advancements represent a substantial step toward developing a reliable early wildfire warning system, which may be our first safeguard against wildfire destruction.

摘要: 最近，迅速蔓延的野火摧毁了社会资产，暴露出迫切需要预警系统来加快救援工作。使用基于相机的深度神经网络（DNN）进行烟雾检测为野火预测提供了一个有前途的解决方案。然而，烟雾在时间和空间中的稀有性限制了训练数据，从而引发了模型过度匹配和偏见的担忧。当前的DNN（主要是卷积神经网络（CNN）和变换器）由于架构差异而使稳健性评估变得复杂。为了应对这些挑战，我们引入了WARP（野火对抗鲁棒性程序），这是第一个用于评估野火检测模型对抗鲁棒性的模型不可知框架。WARP通过图像全局和局部扰动生成对抗性示例来解决数据多样性的固有限制。全局和局部攻击分别将高斯噪音和PNG补丁叠加到图像输入上;这适合CNN和变形器，同时生成现实的对抗场景。使用WARP，我们评估了实时CNN和变形金刚，发现了关键漏洞。有时，在全球攻击下，变压器的精确度会下降超过70%，而这两种模型在局部攻击期间通常很难区分云状的PNG补丁和真正的烟雾。为了增强模型的鲁棒性，我们基于WARP的方法和结果提出了四种面向野火的数据增强技术，这些技术使烟雾图像数据多样化并提高模型精度和鲁棒性。这些进步是朝着开发可靠的早期野火预警系统迈出的重要一步，这可能是我们防止野火破坏的第一个保障。



## **16. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

揭示一致大型语言模型内在的道德脆弱性 cs.CL

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05050v1) [paper-pdf](http://arxiv.org/pdf/2504.05050v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.

摘要: 大型语言模型（LLM）是人工通用智能的基础探索，但它们通过指令调整和偏好学习与人类价值观的一致只能实现表面的合规性。在这里，我们证明，预训练期间嵌入的有害知识在LLM参数记忆中作为不可磨灭的“黑暗模式”持续存在，逃避对齐保障措施，并在分布变化时的对抗诱导下重新浮出水面。在这项研究中，我们首先通过证明当前的对齐方法只产生知识集合中的局部“安全区域”来从理论上分析对齐LLM的内在道德脆弱性。相比之下，预先训练的知识仍然通过高可能性的对抗轨迹与有害概念保持全球联系。基于这一理论见解，我们通过在分布转移下采用语义一致诱导来从经验上验证我们的发现--一种通过优化的对抗提示系统性地绕过对齐约束的方法。这种理论和经验相结合的方法在23个最先进的对齐LLM中的19个（包括DeepSeek-R1和LLaMA-3）上实现了100%的攻击成功率，揭示了它们的普遍漏洞。



## **17. A Domain-Based Taxonomy of Jailbreak Vulnerabilities in Large Language Models**

大型语言模型中基于领域的越狱漏洞分类 cs.CL

21 pages, 5 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04976v1) [paper-pdf](http://arxiv.org/pdf/2504.04976v1)

**Authors**: Carlos Peláez-González, Andrés Herrera-Poyatos, Cristina Zuheros, David Herrera-Poyatos, Virilo Tejedor, Francisco Herrera

**Abstract**: The study of large language models (LLMs) is a key area in open-world machine learning. Although LLMs demonstrate remarkable natural language processing capabilities, they also face several challenges, including consistency issues, hallucinations, and jailbreak vulnerabilities. Jailbreaking refers to the crafting of prompts that bypass alignment safeguards, leading to unsafe outputs that compromise the integrity of LLMs. This work specifically focuses on the challenge of jailbreak vulnerabilities and introduces a novel taxonomy of jailbreak attacks grounded in the training domains of LLMs. It characterizes alignment failures through generalization, objectives, and robustness gaps. Our primary contribution is a perspective on jailbreak, framed through the different linguistic domains that emerge during LLM training and alignment. This viewpoint highlights the limitations of existing approaches and enables us to classify jailbreak attacks on the basis of the underlying model deficiencies they exploit. Unlike conventional classifications that categorize attacks based on prompt construction methods (e.g., prompt templating), our approach provides a deeper understanding of LLM behavior. We introduce a taxonomy with four categories -- mismatched generalization, competing objectives, adversarial robustness, and mixed attacks -- offering insights into the fundamental nature of jailbreak vulnerabilities. Finally, we present key lessons derived from this taxonomic study.

摘要: 大型语言模型（LLM）的研究是开放世界机器学习的一个关键领域。尽管LLM表现出出色的自然语言处理能力，但它们也面临着一些挑战，包括一致性问题、幻觉和越狱漏洞。越狱是指绕过对齐保障措施的提示，导致不安全的输出，从而损害LLM的完整性。这项工作特别关注越狱漏洞的挑战，并引入了一种基于LLM训练领域的新颖越狱攻击分类法。它通过概括性、目标和稳健性差距来描述对齐失败。我们的主要贡献是对越狱的看法，通过LLM培训和调整期间出现的不同语言领域来框架。这一观点强调了现有方法的局限性，并使我们能够根据越狱攻击所利用的基础模型缺陷对越狱攻击进行分类。与基于即时构建方法对攻击进行分类的传统分类不同（例如，提示模板），我们的方法提供了一个更深入的了解LLM行为。我们引入了一个分类法，分为四个类别-不匹配的泛化，竞争目标，对抗性鲁棒性和混合攻击-提供了对越狱漏洞的基本性质的见解。最后，我们提出了从这一分类学研究中得出的关键教训。



## **18. Graph of Effort: Quantifying Risk of AI Usage for Vulnerability Assessment**

努力图表：量化人工智能使用风险以进行漏洞评估 cs.CR

8 pages, 4 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2503.16392v2) [paper-pdf](http://arxiv.org/pdf/2503.16392v2)

**Authors**: Anket Mehra, Andreas Aßmuth, Malte Prieß

**Abstract**: With AI-based software becoming widely available, the risk of exploiting its capabilities, such as high automation and complex pattern recognition, could significantly increase. An AI used offensively to attack non-AI assets is referred to as offensive AI.   Current research explores how offensive AI can be utilized and how its usage can be classified. Additionally, methods for threat modeling are being developed for AI-based assets within organizations. However, there are gaps that need to be addressed. Firstly, there is a need to quantify the factors contributing to the AI threat. Secondly, there is a requirement to create threat models that analyze the risk of being attacked by AI for vulnerability assessment across all assets of an organization. This is particularly crucial and challenging in cloud environments, where sophisticated infrastructure and access control landscapes are prevalent. The ability to quantify and further analyze the threat posed by offensive AI enables analysts to rank vulnerabilities and prioritize the implementation of proactive countermeasures.   To address these gaps, this paper introduces the Graph of Effort, an intuitive, flexible, and effective threat modeling method for analyzing the effort required to use offensive AI for vulnerability exploitation by an adversary. While the threat model is functional and provides valuable support, its design choices need further empirical validation in future work.

摘要: 随着基于人工智能的软件的广泛使用，利用其功能（例如高度自动化和复杂模式识别）的风险可能会显着增加。用于攻击非人工智能资产的人工智能称为进攻性人工智能。   当前的研究探讨了如何利用攻击性人工智能以及如何对其使用进行分类。此外，正在为组织内基于人工智能的资产开发威胁建模方法。然而，也有一些差距需要解决。首先，需要量化导致人工智能威胁的因素。其次，需要创建威胁模型，分析被人工智能攻击的风险，以便对组织所有资产进行漏洞评估。这在复杂的基础设施和访问控制环境普遍存在的云环境中尤其重要和具有挑战性。量化和进一步分析攻击性人工智能构成的威胁的能力使分析师能够对漏洞进行排名并优先考虑主动应对措施的实施。   为了解决这些差距，本文引入了“努力图”，这是一种直观、灵活且有效的威胁建模方法，用于分析对手使用攻击性人工智能进行漏洞利用所需的努力。虽然威胁模型是实用的并提供了有价值的支持，但其设计选择需要在未来的工作中进一步的经验验证。



## **19. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

不要落后，RAG：使用RAG进行免训练对抗检测 cs.AI

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04858v1) [paper-pdf](http://arxiv.org/pdf/2504.04858v1)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.

摘要: 对抗性补丁攻击通过嵌入误导深度模型的局部扰动，对视觉系统构成重大威胁。传统的防御方法通常需要重新培训或微调，这使得它们对于现实世界的部署来说不切实际。我们提出了一个免训练的视觉检索增强生成（VRAG）框架，该框架集成了用于对抗性补丁检测的视觉语言模型（VLM）。通过检索视觉上相似的补丁和图像，这些补丁和图像类似于不断扩展的数据库中存储的攻击，VRAG执行生成式推理以识别不同的攻击类型，而所有这些都无需额外的训练或微调。我们广泛评估了开源大型VLM，包括Qwen-VL-Plus、Qwen2.5-VL-72 B和UI-TARS-72 B-DPO，以及Gemini-2.0（一种闭源模型）。值得注意的是，开源UI-TARS-72 B-DPO模型实现了高达95%的分类准确率，为开源对抗补丁检测奠定了新的最新水平。Gemini-2.0的总体准确率达到了最高的98%，但仍然是闭源的。实验结果证明了VRAG在以最少的人类注释识别各种对抗补丁方面的有效性，为针对不断发展的对抗补丁攻击的稳健、实用的防御铺平了道路。



## **20. Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection**

针对3D资产保护的多视图扩散模型的潜在特征和注意力双重擦除攻击 cs.CV

This paper has been accepted by ICME 2025

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2408.11408v2) [paper-pdf](http://arxiv.org/pdf/2408.11408v2)

**Authors**: Jingwei Sun, Xuchong Zhang, Changfeng Sun, Qicheng Bai, Hongbin Sun

**Abstract**: Multi-View Diffusion Models (MVDMs) enable remarkable improvements in the field of 3D geometric reconstruction, but the issue regarding intellectual property has received increasing attention due to unauthorized imitation. Recently, some works have utilized adversarial attacks to protect copyright. However, all these works focus on single-image generation tasks which only need to consider the inner feature of images. Previous methods are inefficient in attacking MVDMs because they lack the consideration of disrupting the geometric and visual consistency among the generated multi-view images. This paper is the first to address the intellectual property infringement issue arising from MVDMs. Accordingly, we propose a novel latent feature and attention dual erasure attack to disrupt the distribution of latent feature and the consistency across the generated images from multi-view and multi-domain simultaneously. The experiments conducted on SOTA MVDMs indicate that our approach achieves superior performances in terms of attack effectiveness, transferability, and robustness against defense methods. Therefore, this paper provides an efficient solution to protect 3D assets from MVDMs-based 3D geometry reconstruction.

摘要: 多视图扩散模型（MVDM）使3D几何重建领域取得了显着进步，但由于未经授权的模仿，知识产权问题越来越受到关注。最近，一些作品利用对抗攻击来保护版权。然而，所有这些工作都集中在单图像生成任务上，只需要考虑图像的内部特征。以前的方法在攻击MVDM时效率低下，因为它们缺乏考虑破坏生成的多视图图像之间的几何和视觉一致性。本文是第一篇探讨MVDM引起的知识产权侵权问题的论文。因此，我们提出了一种新型的潜在特征和注意力双重擦除攻击，以同时破坏多视图和多域生成图像中潜在特征的分布和一致性。在SOTA MVDM上进行的实验表明，我们的方法在攻击有效性、可转移性和针对防御方法的鲁棒性方面实现了卓越的性能。因此，本文提供了一种有效的解决方案来保护3D资产免受基于MVDM的3D几何重建的影响。



## **21. Towards Benchmarking and Assessing the Safety and Robustness of Autonomous Driving on Safety-critical Scenarios**

在安全关键场景下对自动驾驶的安全性和鲁棒性进行基准测试和评估 cs.RO

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2503.23708v2) [paper-pdf](http://arxiv.org/pdf/2503.23708v2)

**Authors**: Jingzheng Li, Xianglong Liu, Shikui Wei, Zhijun Chen, Bing Li, Qing Guo, Xianqi Yang, Yanjun Pu, Jiakai Wang

**Abstract**: Autonomous driving has made significant progress in both academia and industry, including performance improvements in perception task and the development of end-to-end autonomous driving systems. However, the safety and robustness assessment of autonomous driving has not received sufficient attention. Current evaluations of autonomous driving are typically conducted in natural driving scenarios. However, many accidents often occur in edge cases, also known as safety-critical scenarios. These safety-critical scenarios are difficult to collect, and there is currently no clear definition of what constitutes a safety-critical scenario. In this work, we explore the safety and robustness of autonomous driving in safety-critical scenarios. First, we provide a definition of safety-critical scenarios, including static traffic scenarios such as adversarial attack scenarios and natural distribution shifts, as well as dynamic traffic scenarios such as accident scenarios. Then, we develop an autonomous driving safety testing platform to comprehensively evaluate autonomous driving systems, encompassing not only the assessment of perception modules but also system-level evaluations. Our work systematically constructs a safety verification process for autonomous driving, providing technical support for the industry to establish standardized test framework and reduce risks in real-world road deployment.

摘要: 自动驾驶在学术界和工业界都取得了重大进展，包括感知任务的性能改进和端到端自动驾驶系统的开发。然而，自动驾驶的安全性和稳健性评估尚未得到足够的关注。目前对自动驾驶的评估通常是在自然驾驶场景中进行的。然而，许多事故通常发生在边缘情况下，也称为安全关键情况。这些安全关键场景很难收集，目前还没有明确的定义什么是安全关键场景。在这项工作中，我们探索了自动驾驶在安全关键场景中的安全性和稳健性。首先，我们提供了安全关键场景的定义，包括静态交通场景（例如对抗性攻击场景和自然分布变化）以及动态交通场景（例如事故场景）。然后，我们开发自动驾驶安全测试平台，对自动驾驶系统进行全面评估，不仅包括感知模块的评估，还包括系统级评估。我们的工作系统地构建了自动驾驶的安全验证流程，为行业建立标准化测试框架并降低现实道路部署风险提供技术支持。



## **22. Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models**

两胜一：强大而紧凑的模型的有效集成防御 cs.CV

Accepted to CVPR2025

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04747v1) [paper-pdf](http://arxiv.org/pdf/2504.04747v1)

**Authors**: Yoojin Jung, Byung Cheol Song

**Abstract**: Deep learning-based computer vision systems adopt complex and large architectures to improve performance, yet they face challenges in deployment on resource-constrained mobile and edge devices. To address this issue, model compression techniques such as pruning, quantization, and matrix factorization have been proposed; however, these compressed models are often highly vulnerable to adversarial attacks. We introduce the \textbf{Efficient Ensemble Defense (EED)} technique, which diversifies the compression of a single base model based on different pruning importance scores and enhances ensemble diversity to achieve high adversarial robustness and resource efficiency. EED dynamically determines the number of necessary sub-models during the inference stage, minimizing unnecessary computations while maintaining high robustness. On the CIFAR-10 and SVHN datasets, EED demonstrated state-of-the-art robustness performance compared to existing adversarial pruning techniques, along with an inference speed improvement of up to 1.86 times. This proves that EED is a powerful defense solution in resource-constrained environments.

摘要: 基于深度学习的计算机视觉系统采用复杂且大型的架构来提高性能，但它们在资源有限的移动和边缘设备上部署时面临挑战。为了解决这个问题，人们提出了修剪、量化和矩阵分解等模型压缩技术;然而，这些压缩模型通常极易受到对抗攻击。我们引入了\textBF{高效集合防御（EED）}技术，该技术根据不同的修剪重要性分数使单个基本模型的压缩多样化，并增强集合多样性，以实现高对抗鲁棒性和资源效率。EED在推理阶段动态确定必要子模型的数量，最大限度地减少不必要的计算，同时保持高稳健性。在CIFAR-10和SVHN数据集上，与现有的对抗性修剪技术相比，EED表现出了最先进的鲁棒性性能，并且推理速度提高了高达1.86倍。这证明EED是资源有限环境中强大的防御解决方案。



## **23. A Survey and Evaluation of Adversarial Attacks for Object Detection**

目标检测中的对抗性攻击综述与评价 cs.CV

Accepted for publication in the IEEE Transactions on Neural Networks  and Learning Systems (TNNLS)

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2408.01934v4) [paper-pdf](http://arxiv.org/pdf/2408.01934v4)

**Authors**: Khoi Nguyen Tiet Nguyen, Wenyu Zhang, Kangkang Lu, Yuhuan Wu, Xingjian Zheng, Hui Li Tan, Liangli Zhen

**Abstract**: Deep learning models achieve remarkable accuracy in computer vision tasks, yet remain vulnerable to adversarial examples--carefully crafted perturbations to input images that can deceive these models into making confident but incorrect predictions. This vulnerability pose significant risks in high-stakes applications such as autonomous vehicles, security surveillance, and safety-critical inspection systems. While the existing literature extensively covers adversarial attacks in image classification, comprehensive analyses of such attacks on object detection systems remain limited. This paper presents a novel taxonomic framework for categorizing adversarial attacks specific to object detection architectures, synthesizes existing robustness metrics, and provides a comprehensive empirical evaluation of state-of-the-art attack methodologies on popular object detection models, including both traditional detectors and modern detectors with vision-language pretraining. Through rigorous analysis of open-source attack implementations and their effectiveness across diverse detection architectures, we derive key insights into attack characteristics. Furthermore, we delineate critical research gaps and emerging challenges to guide future investigations in securing object detection systems against adversarial threats. Our findings establish a foundation for developing more robust detection models while highlighting the urgent need for standardized evaluation protocols in this rapidly evolving domain.

摘要: 深度学习模型在计算机视觉任务中实现了非凡的准确性，但仍然容易受到对抗性示例的影响--对输入图像精心设计的扰动，可能会欺骗这些模型做出自信但不正确的预测。该漏洞在自动驾驶汽车、安全监控和安全关键检查系统等高风险应用中构成了重大风险。虽然现有文献广泛涵盖了图像分类中的对抗攻击，但对对象检测系统上的此类攻击的全面分析仍然有限。本文提出了一种新颖的分类框架，用于对特定于对象检测架构的对抗性攻击进行分类，综合了现有的鲁棒性指标，并对流行对象检测模型（包括传统检测器和具有视觉语言预训练的现代检测器）的最新攻击方法进行了全面的实证评估。通过严格分析开源攻击实现及其在不同检测架构中的有效性，我们获得了对攻击特征的关键见解。此外，我们描述了关键的研究差距和新出现的挑战，以指导未来的调查，以确保对象检测系统免受对抗性威胁。我们的研究结果为开发更强大的检测模型奠定了基础，同时强调了在这个快速发展的领域迫切需要标准化的评估协议。



## **24. On the Robustness of GUI Grounding Models Against Image Attacks**

图形用户界面基础模型对抗图像攻击的鲁棒性 cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04716v1) [paper-pdf](http://arxiv.org/pdf/2504.04716v1)

**Authors**: Haoren Zhao, Tianyi Chen, Zhen Wang

**Abstract**: Graphical User Interface (GUI) grounding models are crucial for enabling intelligent agents to understand and interact with complex visual interfaces. However, these models face significant robustness challenges in real-world scenarios due to natural noise and adversarial perturbations, and their robustness remains underexplored. In this study, we systematically evaluate the robustness of state-of-the-art GUI grounding models, such as UGround, under three conditions: natural noise, untargeted adversarial attacks, and targeted adversarial attacks. Our experiments, which were conducted across a wide range of GUI environments, including mobile, desktop, and web interfaces, have clearly demonstrated that GUI grounding models exhibit a high degree of sensitivity to adversarial perturbations and low-resolution conditions. These findings provide valuable insights into the vulnerabilities of GUI grounding models and establish a strong benchmark for future research aimed at enhancing their robustness in practical applications. Our code is available at https://github.com/ZZZhr-1/Robust_GUI_Grounding.

摘要: 图形用户界面（GUI）基础模型对于使智能代理能够理解复杂的可视化界面并与之交互至关重要。然而，由于自然噪声和对抗性扰动，这些模型在现实世界的场景中面临着显著的鲁棒性挑战，并且它们的鲁棒性仍然未得到充分研究。在这项研究中，我们系统地评估了最先进的GUI接地模型（如UGround）在三种条件下的鲁棒性：自然噪声、非针对性对抗攻击和针对性对抗攻击。我们的实验在广泛的GUI环境中进行，包括移动，桌面和Web界面，已经清楚地表明GUI接地模型对对抗性扰动和低分辨率条件表现出高度的敏感性。这些发现为有关图形用户界面基础模型的漏洞提供了宝贵的见解，并为未来旨在增强其在实际应用中稳健性的研究建立了强有力的基准。我们的代码可在https://github.com/ZZZhr-1/Robust_GUI_Grounding上获取。



## **25. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

保护视觉语言模型：缓解基于扰动的攻击中高斯噪音的脆弱性 cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.01308v2) [paper-pdf](http://arxiv.org/pdf/2504.01308v2)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yicheng Fu, Yichun Feng, Kin-Man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.

摘要: 视觉语言模型（VLMS）通过合并视觉信息扩展了大型语言模型（LLM）的功能，但它们仍然容易受到越狱攻击，尤其是在处理嘈杂或损坏的图像时。尽管现有的VLM在培训期间采取安全措施来减轻此类攻击，但与噪音增强视觉输入相关的漏洞被忽视了。在这项工作中，我们发现错过噪音增强训练会导致严重的安全漏洞：许多VLM甚至容易受到高斯噪音等简单扰动的影响。为了应对这一挑战，我们提出了Robust-VLGuard，这是一个具有对齐/未对齐图像-文本对的多模式安全数据集，结合了噪音增强微调，可以降低攻击成功率，同时保留VLM的功能。对于更强的基于优化的视觉扰动攻击，我们提出了DiffPure-VLM，利用扩散模型将对抗性扰动转换为类高斯噪声，可以通过具有噪声增强安全微调的VLM进行防御。实验结果表明，扩散模型的分布偏移特性与我们微调的VLM很好地吻合，显著减轻了不同强度的对抗性扰动。数据集和代码可在https://github.com/JarvisUSTC/DiffPure-RobustVLM上获取。



## **26. Systematic Literature Review on Vehicular Collaborative Perception -- A Computer Vision Perspective**

车辆协作感知的系统文献综述--计算机视觉视角 cs.CV

39 pages, 25 figures

**SubmitDate**: 2025-04-06    [abs](http://arxiv.org/abs/2504.04631v1) [paper-pdf](http://arxiv.org/pdf/2504.04631v1)

**Authors**: Lei Wan, Jianxin Zhao, Andreas Wiedholz, Manuel Bied, Mateus Martinez de Lucena, Abhishek Dinkar Jagtap, Andreas Festag, Antônio Augusto Fröhlich, Hannan Ejaz Keen, Alexey Vinel

**Abstract**: The effectiveness of autonomous vehicles relies on reliable perception capabilities. Despite significant advancements in artificial intelligence and sensor fusion technologies, current single-vehicle perception systems continue to encounter limitations, notably visual occlusions and limited long-range detection capabilities. Collaborative Perception (CP), enabled by Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication, has emerged as a promising solution to mitigate these issues and enhance the reliability of autonomous systems. Beyond advancements in communication, the computer vision community is increasingly focusing on improving vehicular perception through collaborative approaches. However, a systematic literature review that thoroughly examines existing work and reduces subjective bias is still lacking. Such a systematic approach helps identify research gaps, recognize common trends across studies, and inform future research directions. In response, this study follows the PRISMA 2020 guidelines and includes 106 peer-reviewed articles. These publications are analyzed based on modalities, collaboration schemes, and key perception tasks. Through a comparative analysis, this review illustrates how different methods address practical issues such as pose errors, temporal latency, communication constraints, domain shifts, heterogeneity, and adversarial attacks. Furthermore, it critically examines evaluation methodologies, highlighting a misalignment between current metrics and CP's fundamental objectives. By delving into all relevant topics in-depth, this review offers valuable insights into challenges, opportunities, and risks, serving as a reference for advancing research in vehicular collaborative perception.

摘要: 自动驾驶汽车的有效性依赖于可靠的感知能力。尽管人工智能和传感器融合技术取得了重大进步，但当前的单车感知系统继续遇到局限性，特别是视觉遮挡和有限的远程检测能力。由车对车（V2 V）和车对基础设施（V2 I）通信实现的协作感知（CP）已成为缓解这些问题并提高自主系统可靠性的一种有前途的解决方案。除了通信领域的进步之外，计算机视觉界越来越关注通过协作方法改善车辆感知。然而，仍然缺乏彻底审查现有工作并减少主观偏见的系统性文献审查。这种系统性方法有助于识别研究差距、识别研究中的共同趋势，并为未来的研究方向提供信息。作为回应，这项研究遵循PRISMA 2020指南，包括106篇同行评审的文章。这些出版物是根据模式、协作方案和关键感知任务进行分析的。通过比较分析，本综述说明了不同的方法如何解决实际问题，例如姿势错误、时间延迟、通信约束、域转移、异类和对抗性攻击。此外，它还批判性地审查了评估方法，强调了当前指标与CP基本目标之间的不一致。通过深入研究所有相关主题，本评论对挑战、机遇和风险提供了宝贵的见解，为推进车辆协作感知研究提供参考。



## **27. Selective Masking Adversarial Attack on Automatic Speech Recognition Systems**

自动语音识别系统的选择性掩蔽对抗攻击 cs.CR

**SubmitDate**: 2025-04-06    [abs](http://arxiv.org/abs/2504.04394v1) [paper-pdf](http://arxiv.org/pdf/2504.04394v1)

**Authors**: Zheng Fang, Shenyi Zhang, Tao Wang, Bowen Li, Lingchen Zhao, Zhangyi Wang

**Abstract**: Extensive research has shown that Automatic Speech Recognition (ASR) systems are vulnerable to audio adversarial attacks. Current attacks mainly focus on single-source scenarios, ignoring dual-source scenarios where two people are speaking simultaneously. To bridge the gap, we propose a Selective Masking Adversarial attack, namely SMA attack, which ensures that one audio source is selected for recognition while the other audio source is muted in dual-source scenarios. To better adapt to the dual-source scenario, our SMA attack constructs the normal dual-source audio from the muted audio and selected audio. SMA attack initializes the adversarial perturbation with a small Gaussian noise and iteratively optimizes it using a selective masking optimization algorithm. Extensive experiments demonstrate that the SMA attack can generate effective and imperceptible audio adversarial examples in the dual-source scenario, achieving an average success rate of attack of 100% and signal-to-noise ratio of 37.15dB on Conformer-CTC, outperforming the baselines.

摘要: 广泛的研究表明，自动语音识别（ASB）系统容易受到音频对抗攻击。当前的攻击主要集中在单源场景上，忽视了两个人同时说话的双源场景。为了弥合这一差距，我们提出了一种选择性掩蔽对抗攻击，即SM攻击，它可以确保在双源场景中选择一个音频源进行识别，而另一个音频源被静音。为了更好地适应双源场景，我们的SM攻击从静音音频和选定音频中构建正常双源音频。SM攻击用小高斯噪音来掩盖对抗性扰动，并使用选择性掩蔽优化算法对其进行迭代优化。大量实验表明，在双源场景下，SM攻击可以生成有效且难以感知的音频对抗示例，在Conformer-ctc上实现平均攻击成功率为100%，信噪比为37.15分贝，优于基线。



## **28. WeiDetect: Weibull Distribution-Based Defense against Poisoning Attacks in Federated Learning for Network Intrusion Detection Systems**

WeiDetect：网络入侵检测系统联邦学习中基于威布尔分布的中毒攻击防御 cs.CR

**SubmitDate**: 2025-04-06    [abs](http://arxiv.org/abs/2504.04367v1) [paper-pdf](http://arxiv.org/pdf/2504.04367v1)

**Authors**: Sameera K. M., Vinod P., Anderson Rocha, Rafidha Rehiman K. A., Mauro Conti

**Abstract**: In the era of data expansion, ensuring data privacy has become increasingly critical, posing significant challenges to traditional AI-based applications. In addition, the increasing adoption of IoT devices has introduced significant cybersecurity challenges, making traditional Network Intrusion Detection Systems (NIDS) less effective against evolving threats, and privacy concerns and regulatory restrictions limit their deployment. Federated Learning (FL) has emerged as a promising solution, allowing decentralized model training while maintaining data privacy to solve these issues. However, despite implementing privacy-preserving technologies, FL systems remain vulnerable to adversarial attacks. Furthermore, data distribution among clients is not heterogeneous in the FL scenario. We propose WeiDetect, a two-phase, server-side defense mechanism for FL-based NIDS that detects malicious participants to address these challenges. In the first phase, local models are evaluated using a validation dataset to generate validation scores. These scores are then analyzed using a Weibull distribution, identifying and removing malicious models. We conducted experiments to evaluate the effectiveness of our approach in diverse attack settings. Our evaluation included two popular datasets, CIC-Darknet2020 and CSE-CIC-IDS2018, tested under non-IID data distributions. Our findings highlight that WeiDetect outperforms state-of-the-art defense approaches, improving higher target class recall up to 70% and enhancing the global model's F1 score by 1% to 14%.

摘要: 在数据扩展时代，确保数据隐私变得越来越重要，这对传统的基于人工智能的应用构成了重大挑战。此外，物联网设备的日益普及带来了重大的网络安全挑战，使传统的网络入侵检测系统（NIDS）对不断变化的威胁的有效性减弱，隐私问题和监管限制也限制了其部署。联邦学习（FL）已成为一种有前途的解决方案，它允许去中心化模型训练，同时维护数据隐私来解决这些问题。然而，尽管实施了隐私保护技术，FL系统仍然容易受到对抗攻击。此外，在FL场景中，客户端之间的数据分布并不是异类。我们提出了WeiDetect，这是一种用于基于FL的NIDS的两阶段服务器端防御机制，可以检测恶意参与者以解决这些挑战。在第一阶段，使用验证数据集评估本地模型以生成验证分数。然后使用威布尔分布分析这些分数，识别和删除恶意模型。我们进行了实验来评估我们的方法在不同攻击环境中的有效性。我们的评估包括两个流行的数据集：CIC-Darknint 2020和CSE-CIC-IDS 2018，它们在非IID数据分布下进行了测试。我们的研究结果强调，WeiDetect优于最先进的防御方法，将更高的目标类别召回率提高高达70%，并将全球车型的F1得分提高1%至14%。



## **29. Impact of Error Rate Misreporting on Resource Allocation in Multi-tenant Quantum Computing and Defense**

错误率误报对多租户量子计算和国防资源分配的影响 quant-ph

7 pages, 5 figures, conference

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2504.04285v1) [paper-pdf](http://arxiv.org/pdf/2504.04285v1)

**Authors**: Subrata Das, Swaroop Ghosh

**Abstract**: Cloud-based quantum service providers allow multiple users to run programs on shared hardware concurrently to maximize resource utilization and minimize operational costs. This multi-tenant computing (MTC) model relies on the error parameters of the hardware for fair qubit allocation and scheduling, as error-prone qubits can degrade computational accuracy asymmetrically for users sharing the hardware. To maintain low error rates, quantum providers perform periodic hardware calibration, often relying on third-party calibration services. If an adversary within this calibration service misreports error rates, the allocator can be misled into making suboptimal decisions even when the physical hardware remains unchanged. We demonstrate such an attack model in which an adversary strategically misreports qubit error rates to reduce hardware throughput, and probability of successful trial (PST) for two previously proposed allocation frameworks, i.e. Greedy and Community-Based Dynamic Allocation Partitioning (COMDAP). Experimental results show that adversarial misreporting increases execution latency by 24% and reduces PST by 7.8%. We also propose to identify inconsistencies in reported error rates by analyzing statistical deviations in error rates across calibration cycles.

摘要: 基于云的量子服务提供商允许多个用户在共享硬件上同时运行程序，以最大限度地提高资源利用率并最大限度地降低运营成本。这种多租户计算（MT）模型依赖于硬件的错误参数来实现公平的量子位分配和调度，因为容易出错的量子位可能会不对称地降低共享硬件的用户的计算准确性。为了保持低错误率，量子提供商定期执行硬件校准，通常依赖第三方校准服务。如果此校准服务中的对手错误报告了错误率，那么即使物理硬件保持不变，分配器也可能会被误导做出次优决策。我们演示了这样一种攻击模型，其中对手战略性地误报量子位错误率，以降低硬件吞吐量，以及之前提出的两个分配框架（即贪婪和基于社区的动态分配分区（COMDAP））的成功试验概率（BST）。实验结果表明，对抗性误报会使执行延迟增加24%，并使标准时间减少7.8%。我们还建议通过分析校准周期中错误率的统计偏差来识别报告错误率的不一致性。



## **30. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

CyberLLMDirecct：使用网络安全数据分析精调LLM安全性的新数据集 cs.CR

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2503.09334v2) [paper-pdf](http://arxiv.org/pdf/2503.09334v2)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. The dataset creation pipeline, along with comprehensive documentation, examples, and resources for reproducing our results, is publicly available at https://github.com/Adelsamir01/CyberLLMInstruct.

摘要: 将大型语言模型（LLM）集成到网络安全应用程序中带来了重大机会，例如增强威胁分析和恶意软件检测，但也可能带来关键风险和安全问题，包括个人数据泄露和新恶意软件的自动生成。为了应对这些挑战，我们开发了CyberLLMATION，这是一个由54，928个描述-响应对组成的数据集，涵盖网络安全任务，例如恶意软件分析、网络钓鱼模拟和零日漏洞。该数据集是通过多阶段过程构建的。这涉及从多个资源中获取数据，过滤并将其结构化为描述-响应对，并将其与现实世界场景对齐以增强其适用性。选择了七个开源LLM来测试CyberLLMInsurance的有用性：Phi 3 Mini 3.8B、Mistral 7 B、Qwen 2.5 7 B、Llama 3 8B、Llama 3.1 8B、Gemma 2 9 B和Llama 2 70 B。在我们的主要示例中，我们使用OWISP十大框架严格评估了微调模型的安全性，发现微调会降低所有测试的LLM和每次对抗攻击的安全弹性（例如，Llama 3.1 8B对立即注射的安全评分从0.95下降至0.15）。在我们的第二个例子中，我们表明这些相同的微调模型也可以在CyberMetric基准上实现高达92.50%的准确性。这些发现凸显了性能和安全性之间的权衡，表明了对抗性测试和进一步研究微调方法的重要性，这些方法可以降低安全风险，同时仍能提高不同数据集和领域的性能。数据集创建管道以及用于重现我们结果的全面文档、示例和资源可在https://github.com/Adelsamir01/CyberLLMInstruct上公开获取。



## **31. Beating Backdoor Attack at Its Own Game**

在自己的游戏中击败后门攻击 cs.LG

Accepted to ICCV 2023

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2307.15539v4) [paper-pdf](http://arxiv.org/pdf/2307.15539v4)

**Authors**: Min Liu, Alberto Sangiovanni-Vincentelli, Xiangyu Yue

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attack, which does not affect the network's performance on clean data but would manipulate the network behavior once a trigger pattern is added. Existing defense methods have greatly reduced attack success rate, but their prediction accuracy on clean data still lags behind a clean model by a large margin. Inspired by the stealthiness and effectiveness of backdoor attack, we propose a simple but highly effective defense framework which injects non-adversarial backdoors targeting poisoned samples. Following the general steps in backdoor attack, we detect a small set of suspected samples and then apply a poisoning strategy to them. The non-adversarial backdoor, once triggered, suppresses the attacker's backdoor on poisoned data, but has limited influence on clean data. The defense can be carried out during data preprocessing, without any modification to the standard end-to-end training pipeline. We conduct extensive experiments on multiple benchmarks with different architectures and representative attacks. Results demonstrate that our method achieves state-of-the-art defense effectiveness with by far the lowest performance drop on clean data. Considering the surprising defense ability displayed by our framework, we call for more attention to utilizing backdoor for backdoor defense. Code is available at https://github.com/minliu01/non-adversarial_backdoor.

摘要: 深度神经网络（DNN）容易受到后门攻击，后门攻击不会影响网络在干净数据上的性能，但一旦添加触发模式，就会操纵网络行为。现有的防御方法大大降低了攻击成功率，但其对干净数据的预测准确率仍然远远落后于干净模型。受到后门攻击的隐蔽性和有效性的启发，我们提出了一个简单但高效的防御框架，该框架针对有毒样本注入非对抗性后门。遵循后门攻击的一般步骤，我们检测一小组可疑样本，然后对它们应用中毒策略。非对抗性后门一旦被触发，就会抑制攻击者对有毒数据的后门，但对干净数据的影响有限。防御可以在数据预处理期间进行，无需对标准端到端训练管道进行任何修改。我们对具有不同架构和代表性攻击的多个基准进行了广泛的实验。结果表明，我们的方法实现了最先进的防御有效性，并且在干净数据上的性能下降是迄今为止最低的。考虑到我们的框架所表现出的令人惊讶的防御能力，我们呼吁更多地关注利用后门进行后门防御。代码可在https://github.com/minliu01/non-adversarial_backdoor上获取。



## **32. Authenticated Sublinear Quantum Private Information Retrieval**

认证的亚线性量子私有信息检索 quant-ph

11 pages, 1 figure

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2504.04041v1) [paper-pdf](http://arxiv.org/pdf/2504.04041v1)

**Authors**: Fengxia Liu, Zhiyong Zheng, Kun Tian, Yi Zhang, Heng Guo, Zhe Hu, Oleksiy Zhedanov, Zixian Gong

**Abstract**: This paper introduces a novel lower bound on communication complexity using quantum relative entropy and mutual information, refining previous classical entropy-based results. By leveraging Uhlmann's lemma and quantum Pinsker inequalities, the authors establish tighter bounds for information-theoretic security, demonstrating that quantum protocols inherently outperform classical counterparts in balancing privacy and efficiency. Also explores symmetric Quantum Private Information Retrieval (QPIR) protocols that achieve sub-linear communication complexity while ensuring robustness against specious adversaries: A post-quantum cryptography based protocol that can be authenticated for the specious server; A ring-LWE-based protocol for post-quantum security in a single-server setting, ensuring robustness against quantum attacks; A multi-server protocol optimized for hardware practicality, reducing implementation overhead while maintaining sub-linear efficiency. These protocols address critical gaps in secure database queries, offering exponential communication improvements over classical linear-complexity methods. The work also analyzes security trade-offs under quantum specious adversaries, providing theoretical guarantees for privacy and correctness.

摘要: 本文使用量子相对信息引入了一种新颖的通信复杂性下界，完善了之前经典的基于信息量的结果。通过利用乌尔曼引理和量子平斯克不等式，作者为信息论安全建立了更严格的界限，证明量子协议在平衡隐私和效率方面本质上优于经典协议。还探讨了对称量子私有信息检索（QPIR）协议，该协议可以实现次线性通信复杂性，同时确保针对似是而非的对手的鲁棒性：一种基于后量子密码学的协议，可以为似是而非的服务器进行身份验证;一种基于环LWE的协议，用于单服务器设置中的后量子安全，确保针对量子攻击的鲁棒性;针对硬件实用性进行了优化的多服务器协议，在保持次线性效率的同时减少了实施费用。这些协议解决了安全数据库查询中的关键漏洞，提供了比经典线性复杂性方法呈指数级的通信改进。该工作还分析了量子似是而非的对手下的安全权衡，为隐私和正确性提供理论保证。



## **33. Disparate Privacy Vulnerability: Targeted Attribute Inference Attacks and Defenses**

差异隐私漏洞：有针对性的属性推断攻击和防御 cs.LG

Selected for publication at 34th USENIX Security Symposium

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2504.04033v1) [paper-pdf](http://arxiv.org/pdf/2504.04033v1)

**Authors**: Ehsanul Kabir, Lucas Craig, Shagufta Mehnaz

**Abstract**: As machine learning (ML) technologies become more prevalent in privacy-sensitive areas like healthcare and finance, eventually incorporating sensitive information in building data-driven algorithms, it is vital to scrutinize whether these data face any privacy leakage risks. One potential threat arises from an adversary querying trained models using the public, non-sensitive attributes of entities in the training data to infer their private, sensitive attributes, a technique known as the attribute inference attack. This attack is particularly deceptive because, while it may perform poorly in predicting sensitive attributes across the entire dataset, it excels at predicting the sensitive attributes of records from a few vulnerable groups, a phenomenon known as disparate vulnerability. This paper illustrates that an adversary can take advantage of this disparity to carry out a series of new attacks, showcasing a threat level beyond previous imagination. We first develop a novel inference attack called the disparity inference attack, which targets the identification of high-risk groups within the dataset. We then introduce two targeted variations of the attribute inference attack that can identify and exploit a vulnerable subset of the training data, marking the first instances of targeted attacks in this category, achieving significantly higher accuracy than untargeted versions. We are also the first to introduce a novel and effective disparity mitigation technique that simultaneously preserves model performance and prevents any risk of targeted attacks.

摘要: 随着机器学习（ML）技术在医疗保健和金融等隐私敏感领域变得越来越普遍，最终将敏感信息纳入构建数据驱动算法中，审查这些数据是否面临任何隐私泄露风险至关重要。一个潜在的威胁来自对手使用训练数据中实体的公共、非敏感属性来查询训练模型，以推断其私人、敏感属性，这种技术称为属性推断攻击。这种攻击具有特别大的欺骗性，因为虽然它在预测整个数据集中的敏感属性方面可能表现不佳，但它擅长预测少数弱势群体的记录的敏感属性，这种现象被称为不同漏洞。本文说明，对手可以利用这种差异实施一系列新的攻击，展示了超出之前想象的威胁级别。我们首先开发了一种新的推理攻击，称为差异推理攻击，其目标是识别数据集中的高风险群体。然后，我们介绍了两种有针对性的属性推断攻击，可以识别和利用训练数据的脆弱子集，标记这一类有针对性的攻击的第一个实例，实现比非目标版本更高的准确性。我们也是第一个引入一种新颖有效的视差缓解技术，同时保持模型性能并防止任何有针对性的攻击风险。



## **34. Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic Perspective**

了解基于扩散的净化的稳健性：随机视角 cs.CV

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2404.14309v3) [paper-pdf](http://arxiv.org/pdf/2404.14309v3)

**Authors**: Yiming Liu, Kezhao Liu, Yao Xiao, Ziyi Dong, Xiaogang Xu, Pengxu Wei, Liang Lin

**Abstract**: Diffusion-Based Purification (DBP) has emerged as an effective defense mechanism against adversarial attacks. The success of DBP is often attributed to the forward diffusion process, which reduces the distribution gap between clean and adversarial images by adding Gaussian noise. While this explanation is theoretically sound, the exact role of this mechanism in enhancing robustness remains unclear. In this paper, through empirical analysis, we propose that the intrinsic stochasticity in the DBP process is the primary factor driving robustness. To test this hypothesis, we introduce a novel Deterministic White-Box (DW-box) setting to assess robustness in the absence of stochasticity, and we analyze attack trajectories and loss landscapes. Our results suggest that DBP models primarily rely on stochasticity to avoid effective attack directions, while their ability to purify adversarial perturbations may be limited. To further enhance the robustness of DBP models, we propose Adversarial Denoising Diffusion Training (ADDT), which incorporates classifier-guided adversarial perturbations into the diffusion training process, thereby strengthening the models' ability to purify adversarial perturbations. Additionally, we propose Rank-Based Gaussian Mapping (RBGM) to improve the compatibility of perturbations with diffusion models. Experimental results validate the effectiveness of ADDT. In conclusion, our study suggests that future research on DBP can benefit from a clearer distinction between stochasticity-driven and purification-driven robustness.

摘要: 基于扩散的净化（DPP）已成为一种针对对抗性攻击的有效防御机制。DAB的成功通常归因于前向扩散过程，该过程通过添加高斯噪音来缩小干净图像和对抗图像之间的分布差距。虽然这种解释在理论上是合理的，但这种机制在增强稳健性方面的确切作用仍不清楚。本文通过实证分析，提出了CBP过程固有的随机性是驱动鲁棒性的主要因素。为了测试这一假设，我们引入了一种新型的确定性白盒（DW-box）设置来评估在没有随机性的情况下的稳健性，并分析了攻击轨迹和损失格局。我们的结果表明，CBP模型主要依赖随机性来避免有效的攻击方向，而它们净化对抗性扰动的能力可能受到限制。为了进一步增强CBP模型的鲁棒性，我们提出了对抗去噪扩散训练（ADDT），它将分类器引导的对抗性扰动融入到扩散训练过程中，从而加强模型净化对抗性扰动的能力。此外，我们还提出了基于等级的高斯映射（RBGM）来提高扰动与扩散模型的兼容性。实验结果验证了ADDT的有效性。总而言之，我们的研究表明，未来对CBP的研究可以受益于随机性驱动和纯化驱动的稳健性之间的更清晰区分。



## **35. Commit-Reveal$^2$: Randomized Reveal Order Mitigates Last-Revealer Attacks in Commit-Reveal**

Commit-Reveal$#2$：随机显示命令最后缓解-Commit中的显示器攻击-Reveal cs.CR

This paper will appear in the ICBC 2025 proceedings

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03936v1) [paper-pdf](http://arxiv.org/pdf/2504.03936v1)

**Authors**: Suheyon Lee, Euisin Gee

**Abstract**: Randomness generation is a fundamental component in blockchain systems, essential for tasks such as validator selection, zero-knowledge proofs, and decentralized finance operations. Traditional Commit-Reveal mechanisms provide simplicity and security but are susceptible to last revealer attacks, where an adversary can manipulate the random outcome by withholding their reveal. To address this vulnerability, we propose the Commit-Reveal$^2$ protocol, which employs a two-layer Commit-Reveal process to randomize the reveal order and mitigate the risk of such attacks. Additionally, we introduces a method to leverage off-chain networks to optimize communication costs and enhance efficiency. We implement a prototype of the proposed mechanism and publicly release the code to facilitate practical adoption and further research.

摘要: 随机生成是区块链系统中的一个基本组件，对于验证者选择、零知识证明和去中心化财务操作等任务至关重要。传统的Commit-Reveal机制提供了简单性和安全性，但很容易受到最后的揭露者攻击，对手可以通过隐瞒他们的揭露来操纵随机结果。为了解决此漏洞，我们提出了Commit-Reveal$^2$协议，该协议采用两层Commit-Reveal流程来随机化揭示顺序并降低此类攻击的风险。此外，我们还引入了一种利用链下网络来优化通信成本并提高效率的方法。我们实现了拟议机制的原型并公开发布代码，以促进实际采用和进一步研究。



## **36. Towards Adversarially Robust Dataset Distillation by Curvature Regularization**

通过弯曲正规化实现对抗稳健的数据集蒸馏 cs.LG

AAAI 2025

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2403.10045v4) [paper-pdf](http://arxiv.org/pdf/2403.10045v4)

**Authors**: Eric Xue, Yijiang Li, Haoyang Liu, Peiran Wang, Yifan Shen, Haohan Wang

**Abstract**: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information, so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accuracy and robustness with less computation overhead but is also capable of generating robust distilled datasets that can withstand various adversarial attacks. Our implementation is available at: https://github.com/yumozi/GUARD.

摘要: 数据集蒸馏（DD）允许将数据集蒸馏到其原始大小的几分之一，同时保留丰富的分布信息，以便在蒸馏的数据集上训练的模型可以实现相当的准确性，同时节省大量的计算负载。该领域最近的研究一直专注于提高在提炼数据集上训练的模型的准确性。在本文中，我们旨在探索DD的新视角。我们研究如何将对抗鲁棒性嵌入到提取的数据集中，以便在这些数据集上训练的模型保持高准确性，同时获得更好的对抗鲁棒性。我们提出了一种新方法，通过将弯曲正规化融入到蒸馏过程中来实现这一目标，计算费用比标准对抗训练少得多。大量的经验实验表明，我们的方法不仅在准确性和稳健性方面优于标准对抗训练，而且计算负担更少，而且还能够生成能够抵御各种对抗攻击的稳健提取数据集。我们的实现可在：https://github.com/yumozi/GUARD上获取。



## **37. SoK: Attacks on Modern Card Payments**

SoK：对现代卡支付的攻击 cs.CR

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03363v1) [paper-pdf](http://arxiv.org/pdf/2504.03363v1)

**Authors**: Xenia Hofmeier, David Basin, Ralf Sasse, Jorge Toro-Pozo

**Abstract**: EMV is the global standard for smart card payments and its NFC-based version, EMV contactless, is widely used, also for mobile payments. In this systematization of knowledge, we examine attacks on the EMV contactless protocol. We provide a comprehensive framework encompassing its desired security properties and adversary models. We also identify and categorize a comprehensive collection of protocol flaws and show how different subsets thereof can be combined into attacks. In addition to this systematization, we examine the underlying reasons for the many attacks against EMV and point to a better way forward.

摘要: EMV是智能卡支付的全球标准，其基于NFC的版本EMV非接触式版本被广泛使用，也用于移动支付。在知识的系统化中，我们研究了对EMV非接触式协议的攻击。我们提供了一个全面的框架，涵盖其所需的安全属性和对手模型。我们还识别和分类了一系列全面的协议缺陷，并展示了如何将其不同子集组合为攻击。除了这种系统化之外，我们还研究了针对EMV的许多攻击的根本原因，并指出更好的前进道路。



## **38. SLACK: Attacking LiDAR-based SLAM with Adversarial Point Injections**

SLACK：使用对抗点注入攻击基于LiDAR的SLAM cs.CV

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03089v1) [paper-pdf](http://arxiv.org/pdf/2504.03089v1)

**Authors**: Prashant Kumar, Dheeraj Vattikonda, Kshitij Madhav Bhat, Kunal Dargan, Prem Kalra

**Abstract**: The widespread adoption of learning-based methods for the LiDAR makes autonomous vehicles vulnerable to adversarial attacks through adversarial \textit{point injections (PiJ)}. It poses serious security challenges for navigation and map generation. Despite its critical nature, no major work exists that studies learning-based attacks on LiDAR-based SLAM. Our work proposes SLACK, an end-to-end deep generative adversarial model to attack LiDAR scans with several point injections without deteriorating LiDAR quality. To facilitate SLACK, we design a novel yet simple autoencoder that augments contrastive learning with segmentation-based attention for precise reconstructions. SLACK demonstrates superior performance on the task of \textit{point injections (PiJ)} compared to the best baselines on KITTI and CARLA-64 dataset while maintaining accurate scan quality. We qualitatively and quantitatively demonstrate PiJ attacks using a fraction of LiDAR points. It severely degrades navigation and map quality without deteriorating the LiDAR scan quality.

摘要: LiDART广泛采用基于学习的方法，使自动驾驶汽车容易受到通过对抗\textit{point injection（PiJ）}的对抗攻击。它给导航和地图生成带来了严重的安全挑战。尽管其性质至关重要，但目前还没有研究对基于LiDART的SLAM进行基于学习的攻击的主要工作。我们的工作提出了SLACK，这是一种端到端的深度生成对抗模型，可以通过多次点注射攻击LiDART扫描，而不会降低LiDART质量。为了促进SLACK，我们设计了一种新颖而简单的自动编码器，它通过基于分段的注意力增强对比学习，以实现精确重建。与KITTI和CARLA-64数据集的最佳基线相比，SLACK在\textit{点注射（PiJ）}任务中表现出卓越的性能，同时保持准确的扫描质量。我们使用一小部分LiDART点定性和定量地演示了PiJ攻击。它会严重降低导航和地图质量，而不会降低LiDART扫描质量。



## **39. Integrating Identity-Based Identification against Adaptive Adversaries in Federated Learning**

在联邦学习中集成针对自适应对手的基于身份的识别 cs.CR

10 pages, 5 figures, research article, IEEE possible publication (in  submission)

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03077v1) [paper-pdf](http://arxiv.org/pdf/2504.03077v1)

**Authors**: Jakub Kacper Szelag, Ji-Jian Chin, Lauren Ansell, Sook-Chin Yip

**Abstract**: Federated Learning (FL) has recently emerged as a promising paradigm for privacy-preserving, distributed machine learning. However, FL systems face significant security threats, particularly from adaptive adversaries capable of modifying their attack strategies to evade detection. One such threat is the presence of Reconnecting Malicious Clients (RMCs), which exploit FLs open connectivity by reconnecting to the system with modified attack strategies. To address this vulnerability, we propose integration of Identity-Based Identification (IBI) as a security measure within FL environments. By leveraging IBI, we enable FL systems to authenticate clients based on cryptographic identity schemes, effectively preventing previously disconnected malicious clients from re-entering the system. Our approach is implemented using the TNC-IBI (Tan-Ng-Chin) scheme over elliptic curves to ensure computational efficiency, particularly in resource-constrained environments like Internet of Things (IoT). Experimental results demonstrate that integrating IBI with secure aggregation algorithms, such as Krum and Trimmed Mean, significantly improves FL robustness by mitigating the impact of RMCs. We further discuss the broader implications of IBI in FL security, highlighting research directions for adaptive adversary detection, reputation-based mechanisms, and the applicability of identity-based cryptographic frameworks in decentralized FL architectures. Our findings advocate for a holistic approach to FL security, emphasizing the necessity of proactive defence strategies against evolving adaptive adversarial threats.

摘要: 联邦学习（FL）最近已经成为隐私保护，分布式机器学习的一个有前途的范例。然而，FL系统面临着重大的安全威胁，特别是来自能够修改其攻击策略以逃避检测的自适应对手。其中一个威胁是重新连接恶意客户端（RMC）的存在，它通过修改攻击策略重新连接到系统来利用FL开放连接。为了解决这个漏洞，我们建议整合基于身份的识别（IBI）作为FL环境中的安全措施。通过利用IBI，我们使FL系统能够基于加密身份方案对客户端进行身份验证，有效防止之前断开连接的恶意客户端重新进入系统。我们的方法是使用椭圆曲线上的TNC-IBI（Tan-Ng-Chin）方案实施的，以确保计算效率，特别是在物联网（IoT）等资源受限的环境中。实验结果表明，将IBI与Krum和Trimmed Mean等安全聚合算法集成，通过减轻RMC的影响来显着提高FL鲁棒性。我们进一步讨论了IBI在FL安全中的更广泛影响，重点介绍了自适应对手检测、基于声誉的机制以及基于身份的加密框架在去中心化FL架构中的适用性的研究方向。我们的研究结果主张对FL安全采取整体方法，强调针对不断变化的适应性对抗威胁采取积极主动的防御策略的必要性。



## **40. Moving Target Defense Against Adversarial False Data Injection Attacks In Power Grids**

移动目标防御电网中对抗性虚假数据注入攻击 eess.SY

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03065v1) [paper-pdf](http://arxiv.org/pdf/2504.03065v1)

**Authors**: Yexiang Chen, Subhash Lakshminarayana, H. Vincent Poor

**Abstract**: Machine learning (ML)-based detectors have been shown to be effective in detecting stealthy false data injection attacks (FDIAs) that can bypass conventional bad data detectors (BDDs) in power systems. However, ML models are also vulnerable to adversarial attacks. A sophisticated perturbation signal added to the original BDD-bypassing FDIA can conceal the attack from ML-based detectors. In this paper, we develop a moving target defense (MTD) strategy to defend against adversarial FDIAs in power grids. We first develop an MTD-strengthened deep neural network (DNN) model, which deploys a pool of DNN models rather than a single static model that cooperate to detect the adversarial attack jointly. The MTD model pool introduces randomness to the ML model's decision boundary, thereby making the adversarial attacks detectable. Furthermore, to increase the effectiveness of the MTD strategy and reduce the computational costs associated with developing the MTD model pool, we combine this approach with the physics-based MTD, which involves dynamically perturbing the transmission line reactance and retraining the DNN-based detector to adapt to the new system topology. Simulations conducted on IEEE test bus systems demonstrate that the MTD-strengthened DNN achieves up to 94.2% accuracy in detecting adversarial FDIAs. When combined with a physics-based MTD, the detection accuracy surpasses 99%, while significantly reducing the computational costs of updating the DNN models. This approach requires only moderate perturbations to transmission line reactances, resulting in minimal increases in OPF cost.

摘要: 基于机器学习（ML）的检测器已被证明可以有效检测隐蔽的虚假数据注入攻击（FDIA），这些攻击可以绕过电力系统中的传统坏数据检测器（BDS）。然而，ML模型也容易受到对抗攻击。添加到原始BDD旁路FDIA的复杂扰动信号可以隐藏来自基于ML的检测器的攻击。在本文中，我们开发了一种移动目标防御（MTD）策略来防御电网中的敌对FDIA。我们首先开发了一个MTD增强的深度神经网络（DNN）模型，该模型部署了一个DNN模型池，而不是单个静态模型，共同检测对抗性攻击。MTD模型池为ML模型的决策边界引入了随机性，从而使对抗性攻击可检测。此外，为了提高MTD策略的有效性并降低与开发MTD模型池相关的计算成本，我们将这种方法与基于物理的MTD相结合，其中涉及动态扰动传输线阻抗和重新训练基于DNN的检测器以适应新的系统布局。在IEEE测试巴士系统上进行的模拟表明，经过MTD增强的DNN在检测对抗FDIA方面的准确率高达94.2%。与基于物理的MTD相结合时，检测准确率超过99%，同时显着降低了更新DNN模型的计算成本。这种方法只需要对传输线阻抗进行适度的扰动，从而导致OPF成本的增加最小。



## **41. Federated Learning in Adversarial Environments: Testbed Design and Poisoning Resilience in Cybersecurity**

对抗环境中的联邦学习：网络安全中的测试床设计和毒害韧性 cs.CR

6 pages, 4 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2409.09794v2) [paper-pdf](http://arxiv.org/pdf/2409.09794v2)

**Authors**: Hao Jian Huang, Hakan T. Otal, M. Abdullah Canbaz

**Abstract**: This paper presents the design and implementation of a Federated Learning (FL) testbed, focusing on its application in cybersecurity and evaluating its resilience against poisoning attacks. Federated Learning allows multiple clients to collaboratively train a global model while keeping their data decentralized, addressing critical needs for data privacy and security, particularly in sensitive fields like cybersecurity. Our testbed, built using Raspberry Pi and Nvidia Jetson hardware by running the Flower framework, facilitates experimentation with various FL frameworks, assessing their performance, scalability, and ease of integration. Through a case study on federated intrusion detection systems, the testbed's capabilities are shown in detecting anomalies and securing critical infrastructure without exposing sensitive network data. Comprehensive poisoning tests, targeting both model and data integrity, evaluate the system's robustness under adversarial conditions. The results show that while federated learning enhances data privacy and distributed learning, it remains vulnerable to poisoning attacks, which must be mitigated to ensure its reliability in real-world applications.

摘要: 本文介绍了联邦学习（FL）测试平台的设计和实现，重点关注其在网络安全中的应用以及评估其对中毒攻击的弹性。联合学习允许多个客户协作训练全球模型，同时保持数据去中心化，满足数据隐私和安全的关键需求，特别是在网络安全等敏感领域。我们的测试平台使用Raspberry Pi和Nvidia Jetson硬件通过运行Flower框架构建，促进了各种FL框架的实验，评估其性能、可扩展性和集成易用性。通过对联邦入侵检测系统的案例研究，展示了测试平台在检测异常和保护关键基础设施而不暴露敏感网络数据的能力。针对模型和数据完整性的全面中毒测试评估系统在对抗条件下的稳健性。结果表明，虽然联邦学习增强了数据隐私和分布式学习，但它仍然容易受到中毒攻击，必须减轻中毒攻击以确保其在现实世界应用程序中的可靠性。



## **42. ERPO: Advancing Safety Alignment via Ex-Ante Reasoning Preference Optimization**

ERPO：通过前推理偏好优化推进安全一致 cs.CL

18 pages, 5 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02725v1) [paper-pdf](http://arxiv.org/pdf/2504.02725v1)

**Authors**: Kehua Feng, Keyan Ding, Jing Yu, Menghan Li, Yuhao Wang, Tong Xu, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose Ex-Ante Reasoning Preference Optimization (ERPO), a novel safety alignment framework that equips LLMs with explicit preemptive reasoning through Chain-of-Thought and provides clear evidence for safety judgments by embedding predefined safety rules. Specifically, our approach consists of three stages: first, equipping the model with Ex-Ante reasoning through supervised fine-tuning (SFT) using a constructed reasoning module; second, enhancing safety, usefulness, and efficiency via Direct Preference Optimization (DPO); and third, mitigating inference latency with a length-controlled iterative preference optimization strategy. Experiments on multiple open-source LLMs demonstrate that ERPO significantly enhances safety performance while maintaining response efficiency.

摘要: 大型语言模型（LLM）的最新进展加速了人工通用智能的发展，但它们生成有害内容的潜力带来了严重的安全挑战。现有的对齐方法通常难以覆盖各种安全场景，并且仍然容易受到对抗性攻击。在这项工作中，我们提出了前-Ante推理偏好优化（ERPO），一种新的安全对齐框架，通过思想链为LLM提供明确的抢先推理，并通过嵌入预定义的安全规则为安全判断提供明确的证据。具体来说，我们的方法包括三个阶段：第一，通过使用构造的推理模块进行监督微调（SFT），为模型配备Ex-Ante推理;第二，通过直接偏好优化（DPO）提高安全性，有用性和效率;第三，通过长度控制的迭代偏好优化策略减轻推理延迟。在多个开源LLM上的实验表明，ERPO显着增强了安全性能，同时保持了响应效率。



## **43. A Study on Adversarial Robustness of Discriminative Prototypical Learning**

判别式原型学习的对抗鲁棒性研究 cs.LG

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03782v1) [paper-pdf](http://arxiv.org/pdf/2504.03782v1)

**Authors**: Ramin Zarei Sabzevar, Hamed Mohammadzadeh, Tahmineh Tavakoli, Ahad Harati

**Abstract**: Deep neural networks demonstrate significant vulnerability to adversarial perturbations, posing risks for critical applications. Current adversarial training methods predominantly focus on robustness against attacks without explicitly leveraging geometric structures in the latent space, usually resulting in reduced accuracy on the original clean data. To address these issues, we propose a novel adversarial training framework named Adversarial Deep Positive-Negative Prototypes (Adv-DPNP), which integrates disriminative prototype-based learning with adversarial training. Adv-DPNP uses unified class prototypes serving dual roles as classifier weights and robust anchors, enhancing both intra-class compactness and inter-class separation in the latent space. Moreover, a novel dual-branch training mechanism maintains stable prototypes by updating them exclusively with clean data; while the feature extractor layers are learned using both clean and adversarial data to remain invariant against adversarial perturbations. In addition, our approach utilizes a composite loss function combining positive prototype alignment, negative prototype repulsion, and consistency regularization to further enhance discrimination, adversarial robustness, and clean accuracy. Extensive experiments conducted on standard benchmark datasets confirm the effectiveness of Adv-DPNP compared to state-of-the-art methods, achieving higher clean accuracy and competitive robustness under adversarial perturbations and common corruptions. Our code is available at https://github.com/fum-rpl/adv-dpnp

摘要: 深度神经网络表现出对对抗性扰动的严重脆弱性，给关键应用带来风险。当前的对抗训练方法主要关注针对攻击的鲁棒性，而没有明确利用潜在空间中的几何结构，这通常会导致原始干净数据的准确性降低。为了解决这些问题，我们提出了一种新型的对抗性训练框架，名为对抗性深度正-负原型（Adv-DPNP），它将基于区分原型的学习与对抗性训练集成在一起。Adv-DPNP使用统一的类原型，充当分类器权重和稳健锚点的双重角色，增强潜在空间中的类内紧凑性和类间分离性。此外，一种新的双分支训练机制通过专门使用干净数据更新原型来保持稳定的原型;而特征提取器层使用干净数据和对抗数据来学习，以保持对抗扰动的不变。此外，我们的方法利用了一个复合损失函数，结合了正原型对齐、负原型排斥和一致性正则化，以进一步增强区分力、对抗鲁棒性和干净的准确性。在标准基准数据集上进行的大量实验证实了Adv-DPNP与最先进方法相比的有效性，在对抗性扰动和常见腐败下实现了更高的清洁准确性和竞争性鲁棒性。我们的代码可在https://github.com/fum-rpl/adv-dpnp上获取



## **44. No Free Lunch with Guardrails**

没有带护栏的免费午餐 cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.00441v2) [paper-pdf](http://arxiv.org/pdf/2504.00441v2)

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: As large language models (LLMs) and generative AI become widely adopted, guardrails have emerged as a key tool to ensure their safe use. However, adding guardrails isn't without tradeoffs; stronger security measures can reduce usability, while more flexible systems may leave gaps for adversarial attacks. In this work, we explore whether current guardrails effectively prevent misuse while maintaining practical utility. We introduce a framework to evaluate these tradeoffs, measuring how different guardrails balance risk, security, and usability, and build an efficient guardrail.   Our findings confirm that there is no free lunch with guardrails; strengthening security often comes at the cost of usability. To address this, we propose a blueprint for designing better guardrails that minimize risk while maintaining usability. We evaluate various industry guardrails, including Azure Content Safety, Bedrock Guardrails, OpenAI's Moderation API, Guardrails AI, Nemo Guardrails, and Enkrypt AI guardrails. Additionally, we assess how LLMs like GPT-4o, Gemini 2.0-Flash, Claude 3.5-Sonnet, and Mistral Large-Latest respond under different system prompts, including simple prompts, detailed prompts, and detailed prompts with chain-of-thought (CoT) reasoning. Our study provides a clear comparison of how different guardrails perform, highlighting the challenges in balancing security and usability.

摘要: 随着大型语言模型（LLM）和生成式人工智能的广泛采用，护栏已成为确保其安全使用的关键工具。然而，添加护栏并非没有权衡;更强的安全措施可能会降低可用性，而更灵活的系统可能会为对抗性攻击留下缺口。在这项工作中，我们探索当前的护栏是否有效防止滥用，同时保持实用性。我们引入了一个框架来评估这些权衡，衡量不同的护栏如何平衡风险、安全性和可用性，并构建高效的护栏。   我们的调查结果证实，有护栏就没有免费的午餐;加强安全性往往是以牺牲可用性为代价的。为了解决这个问题，我们提出了一个设计更好护栏的蓝图，在保持可用性的同时最大限度地减少风险。我们评估各种行业护栏，包括Azure内容安全、Bedrock Guardrails、OpenAI的Moderation API、Guardrails AI、Nemo Guardrails和Enkrypt AI护栏。此外，我们还评估GPT-4 o、Gemini 2.0-Flash、Claude 3.5-十四行诗和Mistral Large-Latest等LLM如何在不同的系统提示下做出响应，包括简单提示、详细提示和具有思想链（CoT）推理的详细提示。我们的研究对不同护栏的性能进行了清晰的比较，强调了平衡安全性和可用性的挑战。



## **45. Theoretical Insights in Model Inversion Robustness and Conditional Entropy Maximization for Collaborative Inference Systems**

协作推理系统模型倒置稳健性和条件熵最大化的理论见解 cs.LG

accepted by CVPR2025

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2503.00383v2) [paper-pdf](http://arxiv.org/pdf/2503.00383v2)

**Authors**: Song Xia, Yi Yu, Wenhan Yang, Meiwen Ding, Zhuo Chen, Ling-Yu Duan, Alex C. Kot, Xudong Jiang

**Abstract**: By locally encoding raw data into intermediate features, collaborative inference enables end users to leverage powerful deep learning models without exposure of sensitive raw data to cloud servers. However, recent studies have revealed that these intermediate features may not sufficiently preserve privacy, as information can be leaked and raw data can be reconstructed via model inversion attacks (MIAs). Obfuscation-based methods, such as noise corruption, adversarial representation learning, and information filters, enhance the inversion robustness by obfuscating the task-irrelevant redundancy empirically. However, methods for quantifying such redundancy remain elusive, and the explicit mathematical relation between this redundancy minimization and inversion robustness enhancement has not yet been established. To address that, this work first theoretically proves that the conditional entropy of inputs given intermediate features provides a guaranteed lower bound on the reconstruction mean square error (MSE) under any MIA. Then, we derive a differentiable and solvable measure for bounding this conditional entropy based on the Gaussian mixture estimation and propose a conditional entropy maximization (CEM) algorithm to enhance the inversion robustness. Experimental results on four datasets demonstrate the effectiveness and adaptability of our proposed CEM; without compromising feature utility and computing efficiency, plugging the proposed CEM into obfuscation-based defense mechanisms consistently boosts their inversion robustness, achieving average gains ranging from 12.9\% to 48.2\%. Code is available at \href{https://github.com/xiasong0501/CEM}{https://github.com/xiasong0501/CEM}.

摘要: 通过将原始数据本地编码为中间特征，协作推理使最终用户能够利用强大的深度学习模型，而无需将敏感的原始数据暴露给云服务器。然而，最近的研究表明，这些中间特征可能不足以保护隐私，因为信息可能会泄露，并且可以通过模型倒置攻击（MIA）重建原始数据。基于模糊的方法，例如噪音破坏、对抗性表示学习和信息过滤器，通过经验上模糊与任务无关的冗余来增强倒置的鲁棒性。然而，量化此类冗余的方法仍然难以捉摸，并且这种冗余最小化和逆鲁棒性增强之间的明确数学关系尚未建立。为了解决这一问题，这项工作首先从理论上证明，给定中间特征的输入的条件熵为任何MIA下的重建均方误差（SSE）提供了有保证的下限。然后，我们基于高斯混合估计推导出一个可微且可解的方法来限制该条件信息，并提出一种条件信息最大化（MBE）算法来增强逆的鲁棒性。四个数据集的实验结果证明了我们提出的MBE的有效性和适应性;在不损害特征效用和计算效率的情况下，将提出的MBE插入基于模糊的防御机制可持续增强其倒置鲁棒性，实现平均收益范围从12.9%到48.2%。代码可访问\href{https：//github.com/xiasong0501/MBE}{https：//github.com/xiasong0501/MBE}。



## **46. Robust Unsupervised Domain Adaptation for 3D Point Cloud Segmentation Under Source Adversarial Attacks**

源对抗攻击下3D点云分割的鲁棒无监督域自适应 cs.CV

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.01659v2) [paper-pdf](http://arxiv.org/pdf/2504.01659v2)

**Authors**: Haosheng Li, Junjie Chen, Yuecong Xu, Kemi Ding

**Abstract**: Unsupervised domain adaptation (UDA) frameworks have shown good generalization capabilities for 3D point cloud semantic segmentation models on clean data. However, existing works overlook adversarial robustness when the source domain itself is compromised. To comprehensively explore the robustness of the UDA frameworks, we first design a stealthy adversarial point cloud generation attack that can significantly contaminate datasets with only minor perturbations to the point cloud surface. Based on that, we propose a novel dataset, AdvSynLiDAR, comprising synthesized contaminated LiDAR point clouds. With the generated corrupted data, we further develop the Adversarial Adaptation Framework (AAF) as the countermeasure. Specifically, by extending the key point sensitive (KPS) loss towards the Robust Long-Tail loss (RLT loss) and utilizing a decoder branch, our approach enables the model to focus on long-tail classes during the pre-training phase and leverages high-confidence decoded point cloud information to restore point cloud structures during the adaptation phase. We evaluated our AAF method on the AdvSynLiDAR dataset, where the results demonstrate that our AAF method can mitigate performance degradation under source adversarial perturbations for UDA in the 3D point cloud segmentation application.

摘要: 无监督域自适应（UDA）框架对于干净数据上的3D点云语义分割模型表现出良好的概括能力。然而，当源域本身受到损害时，现有作品忽视了对抗稳健性。为了全面探索UDA框架的稳健性，我们首先设计了一种隐形对抗点云生成攻击，该攻击可以严重污染数据集，只需对点云表面进行轻微的扰动。在此基础上，我们提出了一个新型数据集AdvSynLiDART，其中包括合成的污染LiDART点云。利用生成的损坏数据，我们进一步开发对抗性适应框架（AAF）作为对策。具体来说，通过将关键点敏感（KPS）损失扩展到鲁棒长尾损失（RLT损失）并利用解码器分支，我们的方法使模型能够在预训练阶段专注于长尾类，并利用高置信度解码点云信息在适应阶段恢复点云结构。我们在AdvSynLiDART数据集上评估了我们的AAF方法，结果表明我们的AAF方法可以缓解3D点云分割应用程序中UDA源对抗扰动下的性能下降。



## **47. Secure Generalization through Stochastic Bidirectional Parameter Updates Using Dual-Gradient Mechanism**

使用双梯度机制通过随机双向参数更新进行安全推广 cs.LG

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02213v1) [paper-pdf](http://arxiv.org/pdf/2504.02213v1)

**Authors**: Shourya Goel, Himanshi Tibrewal, Anant Jain, Anshul Pundhir, Pravendra Singh

**Abstract**: Federated learning (FL) has gained increasing attention due to privacy-preserving collaborative training on decentralized clients, mitigating the need to upload sensitive data to a central server directly. Nonetheless, recent research has underscored the risk of exposing private data to adversaries, even within FL frameworks. In general, existing methods sacrifice performance while ensuring resistance to privacy leakage in FL. We overcome these issues and generate diverse models at a global server through the proposed stochastic bidirectional parameter update mechanism. Using diverse models, we improved the generalization and feature representation in the FL setup, which also helped to improve the robustness of the model against privacy leakage without hurting the model's utility. We use global models from past FL rounds to follow systematic perturbation in parameter space at the server to ensure model generalization and resistance against privacy attacks. We generate diverse models (in close neighborhoods) for each client by using systematic perturbations in model parameters at a fine-grained level (i.e., altering each convolutional filter across the layers of the model) to improve the generalization and security perspective. We evaluated our proposed approach on four benchmark datasets to validate its superiority. We surpassed the state-of-the-art methods in terms of model utility and robustness towards privacy leakage. We have proven the effectiveness of our method by evaluating performance using several quantitative and qualitative results.

摘要: 由于对去中心化客户端进行保护隐私的协作培训，联合学习（FL）受到了越来越多的关注，减少了将敏感数据直接上传到中央服务器的需要。尽管如此，最近的研究强调了将私人数据暴露给对手的风险，即使在FL框架内也是如此。一般来说，现有方法会牺牲性能，同时确保对FL中隐私泄露的抵抗。我们克服了这些问题，并通过提出的随机双向参数更新机制在全球服务器上生成不同的模型。使用不同的模型，我们改进了FL设置中的概括和特征表示，这也有助于提高模型针对隐私泄露的鲁棒性，而不损害模型的实用性。我们使用过去FL回合的全局模型来跟踪服务器参数空间的系统性扰动，以确保模型的概括性和抵御隐私攻击。我们通过在细粒度级别上使用模型参数的系统性扰动（即，跨模型层改变每个卷积过滤器）以提高概括性和安全性。我们在四个基准数据集上评估了我们提出的方法，以验证其优势。在模型实用性和针对隐私泄露的鲁棒性方面，我们超越了最先进的方法。我们通过使用几个定量和定性结果评估性能来证明了我们方法的有效性。



## **48. FairDAG: Consensus Fairness over Concurrent Causal Design**

FairDAQ：并行因果设计之上的共识公平性 cs.DB

17 pages, 15 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02194v1) [paper-pdf](http://arxiv.org/pdf/2504.02194v1)

**Authors**: Dakai Kang, Junchao Chen, Tien Tuan Anh Dinh, Mohammad Sadoghi

**Abstract**: The rise of cryptocurrencies like Bitcoin and Ethereum has driven interest in blockchain technology, with Ethereum's smart contracts enabling the growth of decentralized finance (DeFi). However, research has shown that adversaries exploit transaction ordering to extract profits through attacks like front-running, sandwich attacks, and liquidation manipulation. This issue affects both permissionless and permissioned blockchains, as block proposers have full control over transaction ordering. To address this, a more fair approach to transaction ordering is essential.   Existing fairness protocols, such as Pompe and Themis, operate on leader-based consensus protocols, which not only suffer from low throughput but also allow adversaries to manipulate transaction ordering. To address these limitations, we propose FairDAG-AB and FairDAG-RL, which leverage DAG-based consensus protocols.   We theoretically demonstrate that FairDAG protocols not only uphold fairness guarantees, as previous fairness protocols do, but also achieve higher throughput and greater resilience to adversarial ordering manipulation. Our deployment and evaluation on CloudLab further validate these claims.

摘要: 比特币和以太坊等加密货币的兴起激发了人们对区块链技术的兴趣，以太坊的智能合约推动了去中心化金融（DeFi）的发展。然而，研究表明，对手利用交易排序通过抢先运行、三明治攻击和清算操纵等攻击来获取利润。这个问题会影响无许可区块链和有许可区块链，因为区块提议者可以完全控制交易排序。为了解决这个问题，更公平的交易排序方法至关重要。   现有的公平协议，例如Pompe和Themis，基于领导者的共识协议运行，该协议不仅吞吐量低，而且允许对手操纵交易排序。为了解决这些限制，我们提出了FairDAG-AB和FairDAG-RL，它们利用基于DAB的共识协议。   我们从理论上证明，FairDAQ协议不仅像以前的公平协议那样坚持公平保证，而且还实现了更高的吞吐量和更大的对抗性排序操纵的弹性。我们在CloudLab上的部署和评估进一步验证了这些说法。



## **49. Learning to Lie: Reinforcement Learning Attacks Damage Human-AI Teams and Teams of LLMs**

学会撒谎：强化学习攻击损害人类人工智能团队和LLM团队 cs.HC

17 pages, 9 figures, accepted to ICLR 2025 Workshop on Human-AI  Coevolution

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2503.21983v2) [paper-pdf](http://arxiv.org/pdf/2503.21983v2)

**Authors**: Abed Kareem Musaffar, Anand Gokhale, Sirui Zeng, Rasta Tadayon, Xifeng Yan, Ambuj Singh, Francesco Bullo

**Abstract**: As artificial intelligence (AI) assistants become more widely adopted in safety-critical domains, it becomes important to develop safeguards against potential failures or adversarial attacks. A key prerequisite to developing these safeguards is understanding the ability of these AI assistants to mislead human teammates. We investigate this attack problem within the context of an intellective strategy game where a team of three humans and one AI assistant collaborate to answer a series of trivia questions. Unbeknownst to the humans, the AI assistant is adversarial. Leveraging techniques from Model-Based Reinforcement Learning (MBRL), the AI assistant learns a model of the humans' trust evolution and uses that model to manipulate the group decision-making process to harm the team. We evaluate two models -- one inspired by literature and the other data-driven -- and find that both can effectively harm the human team. Moreover, we find that in this setting our data-driven model is capable of accurately predicting how human agents appraise their teammates given limited information on prior interactions. Finally, we compare the performance of state-of-the-art LLM models to human agents on our influence allocation task to evaluate whether the LLMs allocate influence similarly to humans or if they are more robust to our attack. These results enhance our understanding of decision-making dynamics in small human-AI teams and lay the foundation for defense strategies.

摘要: 随着人工智能（AI）助手在安全关键领域越来越广泛地采用，开发针对潜在故障或对抗攻击的防护措施变得重要。开发这些保护措施的一个关键先决条件是了解这些人工智能助手误导人类队友的能力。我们在一个推理策略游戏的背景下调查了这个攻击问题，其中三名人类和一名人工智能助理组成的团队合作回答一系列琐碎问题。人类不知道的是，人工智能助手是敌对的。利用基于模型的强化学习（MBRL）的技术，人工智能助手学习人类信任演变的模型，并使用该模型来操纵群体决策过程以伤害团队。我们评估了两个模型--一个受文献启发，另一个受数据驱动--并发现两者都可以有效地伤害人类团队。此外，我们发现，在这种情况下，我们的数据驱动模型能够准确预测人类代理如何在先前互动的有限信息的情况下评估其队友。最后，我们将最先进的LLM模型与人类代理在影响力分配任务中的性能进行了比较，以评估LLM是否以类似于人类的方式分配影响力，或者它们是否对我们的攻击更稳健。这些结果增强了我们对小型人工智能团队决策动态的理解，并为防御策略奠定了基础。



## **50. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

Included in Proceedings of the Conference on Applied Machine Learning  in Information Security (CAMLIS 2024), Arlington, Virginia, USA, October  24-25, 2024

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2406.03230v5) [paper-pdf](http://arxiv.org/pdf/2406.03230v5)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 以OpenAI的ChatGPT为例，大型语言模型（LLM）的广泛采用使防御这些模型上的对抗威胁的必要性变得更加突出。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性以及用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，利用LLM Transformer层之间的剩余激活分析。我们应用一种新颖的方法来分析剩余流中的独特激活模式，以进行攻击提示分类。我们整理了多个数据集，以展示这种分类方法如何在多种类型的攻击场景（包括我们新创建的攻击数据集）中具有高准确性。此外，我们通过集成LLM的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击能力的影响。结果强调了我们的方法在增强对抗性输入的检测和缓解、推进LLC运作的安全框架方面的有效性。




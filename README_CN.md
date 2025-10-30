# Latest Adversarial Attack Papers
**update at 2025-10-30 15:03:12**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Distribution System Reconfiguration to Mitigate Load Altering Attacks via Stackelberg Games**

通过Stackelberg Games重新配置配电系统以减轻负载改变攻击 eess.SY

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2407.07065v6) [paper-pdf](http://arxiv.org/pdf/2407.07065v6)

**Authors**: Sajjad Maleki, E. Veronica Belmaga, Charalambos Konstantinou, Subhash Lakshminarayana

**Abstract**: The widespread integration of IoT-controllable devices (e.g., smart EV charging stations and heat pumps) into modern power systems enhances capabilities but introduces critical cybersecurity risks. Specifically, these devices are susceptible to load-altering attacks (LAAs) that can compromise power system safety. This paper quantifies the impact of LAAs on nodal voltage constraint violations in distribution networks (DNs). We first present closed-form expressions to analytically characterize LAA effects and quantify the minimum number of compromised devices for a successful LAA. Based on these insights, we propose a reactive defense mechanism that mitigates LAAs through DN reconfiguration. To address strategic adversaries, we then formulate defense strategies using a non-cooperative sequential game, which models the knowledgeable and strategic attacker, accounting for the worst-case scenario and enabling the reactive defender to devise an efficient and robust defense. Further, our formulation also accounts for uncertainties in attack localization. A novel Bayesian optimization approach is introduced to compute the Stackelberg equilibrium, significantly reducing computational burden efficiently. The game-theoretic strategy effectively mitigates the attack's impact while ensuring minimal system reconfiguration.

摘要: 物联网可控设备的广泛集成（例如，智能电动汽车充电站和热泵）融入现代电力系统可以增强功能，但也会带来严重的网络安全风险。具体来说，这些设备很容易受到负载改变攻击（LAA），从而危及电力系统安全。本文量化了LAA对配电网（DN）中节点电压约束违规的影响。我们首先提出封闭形式的表达来分析描述LAA效应，并量化成功LAA的受损设备的最少数量。基于这些见解，我们提出了一种反应式防御机制，通过DN重新配置减轻LAA。为了应对战略对手，我们然后使用非合作顺序博弈来制定防御策略，该博弈对知识渊博且具有战略意义的攻击者进行建模，考虑到最坏的情况，并使反应型防御者能够设计出高效且稳健的防御。此外，我们的公式还考虑了攻击定位的不确定性。引入了一种新颖的Bayesian优化方法来计算Stackelberg均衡，有效地显着减少了计算负担。博弈论策略有效地减轻了攻击的影响，同时确保最少的系统重新配置。



## **2. Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation**

鲍勃的五彩纸屑：音乐和视频生成中的语音同步攻击 cs.SD

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2507.17937v3) [paper-pdf](http://arxiv.org/pdf/2507.17937v3)

**Authors**: Jaechul Roh, Zachary Novack, Yuefeng Peng, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Amir Houmansadr

**Abstract**: Generative AI systems for music and video commonly use text-based filters to prevent the regurgitation of copyrighted material. We expose a fundamental flaw in this approach by introducing Adversarial PhoneTic Prompting (APT), a novel attack that bypasses these safeguards by exploiting phonetic memorization. The APT attack replaces iconic lyrics with homophonic but semantically unrelated alternatives (e.g., "mom's spaghetti" becomes "Bob's confetti"), preserving acoustic structure while altering meaning; we identify high-fidelity phonetic matches using CMU pronouncing dictionary. We demonstrate that leading Lyrics-to-Song (L2S) models like SUNO and YuE regenerate songs with striking melodic and rhythmic similarity to their copyrighted originals when prompted with these altered lyrics. More surprisingly, this vulnerability extends across modalities. When prompted with phonetically modified lyrics from a song, a Text-to-Video (T2V) model like Veo 3 reconstructs visual scenes from the original music video-including specific settings and character archetypes-despite the absence of any visual cues in the prompt. Our findings reveal that models memorize deep, structural patterns tied to acoustics, not just verbatim text. This phonetic-to-visual leakage represents a critical vulnerability in transcript-conditioned generative models, rendering simple copyright filters ineffective and raising urgent concerns about the secure deployment of multimodal AI systems. Demo examples are available at our project page (https://jrohsc.github.io/music_attack/).

摘要: 音乐和视频的生成人工智能系统通常使用基于文本的过滤器来防止受版权保护的材料的回流。我们通过引入对抗音素提取（APT）来暴露这种方法的一个根本缺陷，APT是一种通过利用语音记忆绕过这些保障措施的新型攻击。APT攻击用谐音但语义无关的替代方案取代标志性歌词（例如，“妈妈的意大利面”变成了“鲍勃的五彩纸屑”），在改变含义的同时保留了声学结构;我们使用CMU发音词典识别高保真语音匹配。我们证明，当提示这些修改后的歌词时，SUNO和YuE等领先的歌词转歌曲（L2 S）模型会以与受版权保护的原创歌曲具有惊人的旋律和节奏相似性。更令人惊讶的是，这种脆弱性延伸到各个模式中。当提示歌曲中经过语音修改的歌词时，Veo 3等文本转视频（T2 V）模型会从原始音乐视频中重建视觉场景-包括特定的设置和角色原型-尽管提示中没有任何视觉提示。我们的研究结果表明，模型会记住与声学相关的深层结构模式，而不仅仅是逐字逐句的文本。这种语音到视觉的泄露代表了转录条件生成模型中的一个关键漏洞，使简单的版权过滤器无效，并引发了对多模式人工智能系统安全部署的紧迫担忧。演示示例可在我们的项目页面（https：//jrohsc.github.io/music_attack/）上找到。



## **3. NetEcho: From Real-World Streaming Side-Channels to Full LLM Conversation Recovery**

NetEcho：从现实世界的流媒体副频道到完整的LLM对话恢复 cs.CR

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.25472v1) [paper-pdf](http://arxiv.org/pdf/2510.25472v1)

**Authors**: Zheng Zhang, Guanlong Wu, Sen Deng, Shuai Wang, Yinqian Zhang

**Abstract**: In the rapidly expanding landscape of Large Language Model (LLM) applications, real-time output streaming has become the dominant interaction paradigm. While this enhances user experience, recent research reveals that it exposes a non-trivial attack surface through network side-channels. Adversaries can exploit patterns in encrypted traffic to infer sensitive information and reconstruct private conversations. In response, LLM providers and third-party services are deploying defenses such as traffic padding and obfuscation to mitigate these vulnerabilities.   This paper starts by presenting a systematic analysis of contemporary side-channel defenses in mainstream LLM applications, with a focus on services from vendors like OpenAI and DeepSeek. We identify and examine seven representative deployment scenarios, each incorporating active/passive mitigation techniques. Despite these enhanced security measures, our investigation uncovers significant residual information that remains vulnerable to leakage within the network traffic.   Building on this discovery, we introduce NetEcho, a novel, LLM-based framework that comprehensively unleashes the network side-channel risks of today's LLM applications. NetEcho is designed to recover entire conversations -- including both user prompts and LLM responses -- directly from encrypted network traffic. It features a deliberate design that ensures high-fidelity text recovery, transferability across different deployment scenarios, and moderate operational cost. In our evaluations on medical and legal applications built upon leading models like DeepSeek-v3 and GPT-4o, NetEcho can recover avg $\sim$70\% information of each conversation, demonstrating a critical limitation in current defense mechanisms. We conclude by discussing the implications of our findings and proposing future directions for augmenting network traffic security.

摘要: 在大型语言模型（LLM）应用程序的迅速扩大中，实时输出流已成为主导的交互范式。虽然这增强了用户体验，但最近的研究表明，它通过网络侧渠道暴露了一个不平凡的攻击表面。对手可以利用加密流量中的模式来推断敏感信息并重建私人对话。作为回应，LLM提供商和第三方服务正在部署流量填充和混淆等防御措施来缓解这些漏洞。   本文首先对主流LLM应用程序中的当代侧通道防御进行了系统分析，重点关注OpenAI和DeepSeek等供应商的服务。我们确定并检查了七种代表性的部署场景，每种场景都结合了主动/被动缓解技术。尽管采取了这些增强的安全措施，我们的调查仍发现了大量残留信息，这些信息仍然容易在网络流量中泄露。   在这一发现的基础上，我们引入了NetEcho，这是一个基于LLM的新颖框架，可以全面释放当今LLM应用程序的网络侧通道风险。NetEcho旨在直接从加密的网络流量中恢复整个对话（包括用户提示和LLM响应）。它采用精心设计的设计，确保高保真文本恢复、不同部署场景之间的可移植性以及适度的运营成本。在我们对基于DeepSeek-v3和GPT-4 o等领先模型的医疗和法律应用程序的评估中，NetEcho可以恢复每次对话的平均$70%信息，这表明了当前防御机制的严重局限性。最后，我们讨论了我们研究结果的影响，并提出了增强网络流量安全的未来方向。



## **4. Timestamp Manipulation: Timestamp-based Nakamoto-style Blockchains are Vulnerable**

时间戳操纵：基于时间戳的中本风格区块链很脆弱 cs.CR

25 pages, 6 figures

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2505.05328v5) [paper-pdf](http://arxiv.org/pdf/2505.05328v5)

**Authors**: Junjie Hu, Sisi Duan

**Abstract**: Nakamoto consensus are the most widely adopted decentralized consensus mechanism in cryptocurrency systems. Since it was proposed in 2008, many studies have focused on analyzing its security. Most of them focus on maximizing the profit of the adversary. Examples include the selfish mining attack [FC '14] and the recent riskless uncle maker (RUM) attack [CCS '23]. In this work, we introduce the Staircase-Unrestricted Uncle Maker (SUUM), the first block withholding attack targeting the timestamp-based Nakamoto-style blockchain. Through block withholding, timestamp manipulation, and difficulty risk control, SUUM adversaries are capable of launching persistent attacks with zero cost and minimal difficulty risk characteristics, indefinitely exploiting rewards from honest participants. This creates a self-reinforcing cycle that threatens the security of blockchains. We conduct a comprehensive and systematic evaluation of SUUM, including the attack conditions, its impact on blockchains, and the difficulty risks. Finally, we further discuss four feasible mitigation measures against SUUM.

摘要: Nakamoto共识是加密货币系统中最广泛采用的去中心化共识机制。自2008年提出以来，许多研究都集中在分析其安全性上。他们中的大多数都专注于使对手的利润最大化。例子包括自私的采矿攻击[FC ' 14]和最近的无风险叔叔制造商（RUM）攻击[CS ' 23]。在这项工作中，我们介绍了Staircase-Unrestricted Uncle Maker（SUUM），这是针对基于时间戳的Nakamoto风格区块链的第一个阻止攻击。通过区块扣留、时间戳操纵和难度风险控制，SUUM对手能够以零成本和最小难度风险特征发起持续攻击，无限期地利用诚实参与者的回报。这造成了一个自我强化的循环，威胁区块链的安全。我们对SUUM进行了全面系统的评估，包括攻击条件，对区块链的影响以及难度风险。最后，我们进一步讨论了四个可行的缓解措施，对SUUM。



## **5. TextCrafter: Optimization-Calibrated Noise for Defending Against Text Embedding Inversion**

TextCrafter：用于防御文本嵌入倒置的优化校准噪音 cs.CR

More sufficient and convincing experiments are needed

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2509.17302v2) [paper-pdf](http://arxiv.org/pdf/2509.17302v2)

**Authors**: Duoxun Tang, Xinhang Jiang, Jiajun Niu

**Abstract**: Text embedding inversion attacks reconstruct original sentences from latent representations, posing severe privacy threats in collaborative inference and edge computing. We propose TextCrafter, an optimization-based adversarial perturbation mechanism that combines RL learned, geometry aware noise injection orthogonal to user embeddings with cluster priors and PII signal guidance to suppress inversion while preserving task utility. Unlike prior defenses either non learnable or agnostic to perturbation direction, TextCrafter provides a directional protective policy that balances privacy and utility. Under strong privacy setting, TextCrafter maintains 70 percentage classification accuracy on four datasets and consistently outperforms Gaussian/LDP baselines across lower privacy budgets, demonstrating a superior privacy utility trade off.

摘要: 文本嵌入倒置攻击从潜在表示中重建原始句子，在协作推理和边缘计算中构成严重的隐私威胁。我们提出了TextCrafter，这是一种基于优化的对抗性扰动机制，它将RL学习的、与用户嵌入垂直的几何感知噪音注入与集群先验和PRI信号引导相结合，以抑制倒置，同时保留任务效用。与之前的防御（无论是不可学习的还是对扰动方向不可知的）不同，TextCrafter提供了一种平衡隐私和实用性的定向保护政策。在强大的隐私设置下，TextCrafter在四个数据集上保持了70%的分类准确性，并且在较低的隐私预算下始终优于高斯/LDP基线，展示了卓越的隐私实用权衡。



## **6. A Unified Bilevel Model for Adversarial Learning and A Case Study**

对抗学习的统一二层模型及案例研究 cs.LG

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.25121v1) [paper-pdf](http://arxiv.org/pdf/2510.25121v1)

**Authors**: Yutong Zheng, Qingna Li

**Abstract**: Adversarial learning has been attracting more and more attention thanks to the fast development of machine learning and artificial intelligence. However, due to the complicated structure of most machine learning models, the mechanism of adversarial attacks is not well interpreted. How to measure the effect of attack is still not quite clear. In this paper, we propose a unified bilevel model for adversarial learning. We further investigate the adversarial attack in clustering models and interpret it from data perturbation point of view. We reveal that when the data perturbation is relatively small, the clustering model is robust, whereas if it is relatively large, the clustering result changes, which leads to an attack. To measure the effect of attacks for clustering models, we analyse the well-definedness of the so-called $\delta$-measure, which can be used in the proposed bilevel model for adversarial learning of clustering models.

摘要: 随着机器学习和人工智能的快速发展，对抗性学习越来越受到关注。然而，由于大多数机器学习模型的结构复杂，对抗性攻击的机制没有得到很好的解释。如何衡量攻击的效果还不太清楚。在本文中，我们提出了一个统一的对抗学习二层模型。我们进一步研究了集群模型中的对抗攻击，并从数据扰动的角度对其进行解释。我们发现，当数据扰动相对较小时，集群模型是稳健的，而如果数据扰动相对较大，集群结果就会发生变化，从而导致攻击。为了衡量攻击对集群模型的影响，我们分析了所谓的$\delta$-测量的明确性，该测量可用于提出的二层模型，用于集群模型的对抗学习。



## **7. An Adversarial-Driven Experimental Study on Deep Learning for RF Fingerprinting**

对抗驱动的RF指纹深度学习实验研究 cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2507.14109v2) [paper-pdf](http://arxiv.org/pdf/2507.14109v2)

**Authors**: Xinyu Cao, Bimal Adhikari, Shangqing Zhao, Jingxian Wu, Yanjun Pan

**Abstract**: Radio frequency (RF) fingerprinting, which extracts unique hardware imperfections of radio devices, has emerged as a promising physical-layer device identification mechanism in zero trust architectures and beyond 5G networks. In particular, deep learning (DL) methods have demonstrated state-of-the-art performance in this domain. However, existing approaches have primarily focused on enhancing system robustness against temporal and spatial variations in wireless environments, while the security vulnerabilities of these DL-based approaches have often been overlooked. In this work, we systematically investigate the security risks of DL-based RF fingerprinting systems through an adversarial-driven experimental analysis. We observe a consistent misclassification behavior for DL models under domain shifts, where a device is frequently misclassified as another specific one. Our analysis based on extensive real-world experiments demonstrates that this behavior can be exploited as an effective backdoor to enable external attackers to intrude into the system. Furthermore, we show that training DL models on raw received signals causes the models to entangle RF fingerprints with environmental and signal-pattern features, creating additional attack vectors that cannot be mitigated solely through post-processing security methods such as confidence thresholds.

摘要: 射频（RF）指纹识别提取无线电设备独特的硬件缺陷，已成为零信任架构和5G网络以外的一种有前途的物理层设备识别机制。特别是，深度学习（DL）方法在该领域展示了最先进的性能。然而，现有方法主要集中在增强系统针对无线环境中的时间和空间变化的鲁棒性，而这些基于DL的方法的安全漏洞常常被忽视。在这项工作中，我们通过对抗驱动的实验分析系统地调查了基于DL的RF指纹识别系统的安全风险。我们观察到DL模型在域转移下存在一致的错误分类行为，其中设备经常被错误分类为另一个特定设备。我们基于广泛的现实实验的分析表明，这种行为可以被利用为有效的后门，使外部攻击者能够入侵系统。此外，我们表明，在原始接收信号上训练DL模型会导致模型将RF指纹与环境和信号模式特征纠缠在一起，从而创建额外的攻击载体，这些攻击载体无法仅通过置信阈值等后处理安全方法来缓解。



## **8. Jailbreak Transferability Emerges from Shared Representations**

越狱可转让性源于共享表示 cs.LG

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2506.12913v2) [paper-pdf](http://arxiv.org/pdf/2506.12913v2)

**Authors**: Rico Angell, Jannik Brinkmann, He He

**Abstract**: Jailbreak transferability is the surprising phenomenon when an adversarial attack compromising one model also elicits harmful responses from other models. Despite widespread demonstrations, there is little consensus on why transfer is possible: is it a quirk of safety training, an artifact of model families, or a more fundamental property of representation learning? We present evidence that transferability emerges from shared representations rather than incidental flaws. Across 20 open-weight models and 33 jailbreak attacks, we find two factors that systematically shape transfer: (1) representational similarity under benign prompts, and (2) the strength of the jailbreak on the source model. To move beyond correlation, we show that deliberately increasing similarity through benign only distillation causally increases transfer. Our qualitative analyses reveal systematic transferability patterns across different types of jailbreaks. For example, persona-style jailbreaks transfer far more often than cipher-based prompts, consistent with the idea that natural-language attacks exploit models' shared representation space, whereas cipher-based attacks rely on idiosyncratic quirks that do not generalize. Together, these results reframe jailbreak transfer as a consequence of representation alignment rather than a fragile byproduct of safety training.

摘要: 当损害一个模型的对抗攻击同时也会引发其他模型的有害反应时，越狱可移植性是一种令人惊讶的现象。尽管进行了广泛的演示，但对于为什么可以转移几乎没有达成共识：这是安全培训的怪癖、模范家庭的产物，还是代表学习的更基本属性？我们提供的证据表明，可转让性源于共同的表示，而不是偶然的缺陷。在20个开放权重模型和33次越狱攻击中，我们发现了两个系统性影响转移的因素：（1）良性提示下的代表相似性，（2）越狱对源模型的强度。为了超越相关性，我们表明通过良性蒸馏故意增加相似性会导致转移。我们的定性分析揭示了不同类型越狱的系统性转移模式。例如，人物风格的越狱传输频率远高于基于密码的提示，这与自然语言攻击利用模型的共享表示空间的观点一致，而基于密码的攻击依赖于不概括的独特怪癖。总而言之，这些结果将越狱转移重新定义为代表一致的结果，而不是安全培训的脆弱副产品。



## **9. Cybersecurity AI Benchmark (CAIBench): A Meta-Benchmark for Evaluating Cybersecurity AI Agents**

网络安全人工智能基准（CAIBench）：评估网络安全人工智能代理的元基准 cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24317v1) [paper-pdf](http://arxiv.org/pdf/2510.24317v1)

**Authors**: María Sanz-Gómez, Víctor Mayoral-Vilches, Francesco Balassone, Luis Javier Navarrete-Lozano, Cristóbal R. J. Veas Chavez, Maite del Mundo de Torres

**Abstract**: Cybersecurity spans multiple interconnected domains, complicating the development of meaningful, labor-relevant benchmarks. Existing benchmarks assess isolated skills rather than integrated performance. We find that pre-trained knowledge of cybersecurity in LLMs does not imply attack and defense abilities, revealing a gap between knowledge and capability. To address this limitation, we present the Cybersecurity AI Benchmark (CAIBench), a modular meta-benchmark framework that allows evaluating LLM models and agents across offensive and defensive cybersecurity domains, taking a step towards meaningfully measuring their labor-relevance. CAIBench integrates five evaluation categories, covering over 10,000 instances: Jeopardy-style CTFs, Attack and Defense CTFs, Cyber Range exercises, knowledge benchmarks, and privacy assessments. Key novel contributions include systematic simultaneous offensive-defensive evaluation, robotics-focused cybersecurity challenges (RCTF2), and privacy-preserving performance assessment (CyberPII-Bench). Evaluation of state-of-the-art AI models reveals saturation on security knowledge metrics (~70\% success) but substantial degradation in multi-step adversarial (A\&D) scenarios (20-40\% success), or worse in robotic targets (22\% success). The combination of framework scaffolding and LLM model choice significantly impacts performance; we find that proper matches improve up to 2.6$\times$ variance in Attack and Defense CTFs. These results demonstrate a pronounced gap between conceptual knowledge and adaptive capability, emphasizing the need for a meta-benchmark.

摘要: 网络安全跨越多个相互关联的领域，使有意义的、与劳动力相关的基准的开发变得复杂。现有的基准评估孤立的技能而不是综合的绩效。我们发现，LLM中预先训练的网络安全知识并不意味着攻击和防御能力，这揭示了知识和能力之间的差距。为了解决这一局限性，我们提出了网络安全人工智能基准（CAIBench），这是一个模块化元基准框架，允许评估跨进攻性和防御性网络安全领域的LLM模型和代理，朝着有意义地衡量其劳动相关性迈出了一步。CAIBench集成了五个评估类别，涵盖10，000多个实例：《危险边缘》风格的CTF、攻击和防御CTF、网络范围练习、知识基准和隐私评估。主要的创新贡献包括系统性的同时进攻-防御评估、以机器人为中心的网络安全挑战（RCTF 2）和隐私保护性能评估（CyberPII-Bench）。对最先进人工智能模型的评估显示，安全知识指标已饱和（成功率~ 70%），但在多步对抗（A & D）场景中大幅下降（成功率为20- 40%），或者在机器人目标中更糟（成功率为22%）。框架脚手架和LLM模型选择的结合显着影响性能;我们发现适当的匹配可以提高攻击和防御CTF中高达2.6 $\times $方差。这些结果表明概念知识和适应能力之间存在明显差距，强调了元基准的必要性。



## **10. MixAT: Combining Continuous and Discrete Adversarial Training for LLMs**

MixAT：结合LLM的连续和离散对抗训练 cs.LG

Published at 39th Conference on Neural Information Processing Systems  (NeurIPS 2025)

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2505.16947v2) [paper-pdf](http://arxiv.org/pdf/2505.16947v2)

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: Despite recent efforts in Large Language Model (LLM) safety and alignment, current adversarial attacks on frontier LLMs can still consistently force harmful generations. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. At the same time, despite their effectiveness and generalization capabilities, training with continuous perturbations does not always capture the full spectrum of vulnerabilities exploited by discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.

摘要: 尽管最近在大型语言模型（LLM）的安全性和对齐方面做出了努力，但目前对前沿LLM的对抗性攻击仍然可以持续地迫使有害的世代。尽管对抗性训练已经被广泛研究，并被证明可以显着提高传统机器学习模型的鲁棒性，但它在LLM背景下的优势和劣势却鲜为人知。具体来说，虽然现有的离散对抗性攻击在产生有害内容方面是有效的，但用具体的对抗性提示训练LLM通常在计算上是昂贵的，导致对连续松弛的依赖。与此同时，尽管具有有效性和泛化能力，但连续扰动训练并不总是能够捕获离散攻击所利用的全部漏洞。在这项工作中，我们的目标是通过引入MixAT来弥合这一差距，MixAT是一种新颖的方法，在训练期间结合了更强的离散攻击和更快的连续攻击。我们对MixAT进行了广泛的最先进攻击，提出了至少一次攻击成功率（ALO-ASB）指标来捕捉模型的最坏情况漏洞。我们表明，与之前的防御（ALO-ASB> 50%）相比，MixAT实现了更好的鲁棒性（ALO-ASB < 20%），同时保持与基于连续松弛的方法相当的运行时间。我们进一步分析了现实部署环境中的MixAT，探索聊天模板、量化、低等级适配器和温度如何影响对抗训练和评估，从而揭示了当前方法中的其他盲点。我们的结果表明，MixAT的离散-连续防御以最小的计算负担提供了原则性且卓越的鲁棒性-准确性权衡，凸显了其构建更安全的LLM的承诺。我们在https://github.com/insait-institute/MixAT上提供我们的代码和模型。



## **11. Is It Certainly a Deepfake? Reliability Analysis in Detection & Generation Ecosystem**

难道真的是假的吗？检测与发电生态系统中的可靠性分析 cs.AI

Accepted for publication at the ICCV 2025 workshop - STREAM

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2509.17550v3) [paper-pdf](http://arxiv.org/pdf/2509.17550v3)

**Authors**: Neslihan Kose, Anthony Rhodes, Umur Aybars Ciftci, Ilke Demir

**Abstract**: As generative models are advancing in quality and quantity for creating synthetic content, deepfakes begin to cause online mistrust. Deepfake detectors are proposed to counter this effect, however, misuse of detectors claiming fake content as real or vice versa further fuels this misinformation problem. We present the first comprehensive uncertainty analysis of deepfake detectors, systematically investigating how generative artifacts influence prediction confidence. As reflected in detectors' responses, deepfake generators also contribute to this uncertainty as their generative residues vary, so we cross the uncertainty analysis of deepfake detectors and generators. Based on our observations, the uncertainty manifold holds enough consistent information to leverage uncertainty for deepfake source detection. Our approach leverages Bayesian Neural Networks and Monte Carlo dropout to quantify both aleatoric and epistemic uncertainties across diverse detector architectures. We evaluate uncertainty on two datasets with nine generators, with four blind and two biological detectors, compare different uncertainty methods, explore region- and pixel-based uncertainty, and conduct ablation studies. We conduct and analyze binary real/fake, multi-class real/fake, source detection, and leave-one-out experiments between the generator/detector combinations to share their generalization capability, model calibration, uncertainty, and robustness against adversarial attacks. We further introduce uncertainty maps that localize prediction confidence at the pixel level, revealing distinct patterns correlated with generator-specific artifacts. Our analysis provides critical insights for deploying reliable deepfake detection systems and establishes uncertainty quantification as a fundamental requirement for trustworthy synthetic media detection.

摘要: 随着生成模型在创建合成内容的质量和数量上不断进步，深度造假开始引起在线不信任。Deepfake检测器被提出来对抗这种影响，然而，滥用检测器将虚假内容称为真实内容或反之亦然，进一步加剧了这种错误信息问题。我们首次对Deepfake检测器进行了全面的不确定性分析，系统地研究生成伪影如何影响预测置信度。正如检测器的响应所反映的那样，Deepfake生成器也会导致这种不确定性，因为它们的生成残余有所不同，因此我们交叉了Deepfake检测器和生成器的不确定性分析。根据我们的观察，不确定性流形拥有足够的一致信息，可以利用不确定性进行深度伪造源检测。我们的方法利用Bayesian神经网络和Monte Carlo dropout来量化不同检测器架构中的任意性和认识性不确定性。我们评估了两个数据集的不确定性，该数据集具有九个发生器、四个盲探测器和两个生物探测器，比较不同的不确定性方法，探索基于区域和像素的不确定性，并进行消融研究。我们在生成器/检测器组合之间进行并分析二进制实/假、多类实/假、源检测和留一实验，以分享它们的概括能力、模型校准、不确定性和对抗性攻击的鲁棒性。我们进一步引入不确定性地图，在像素级定位预测的信心，揭示了不同的模式与发电机特定的文物。我们的分析为部署可靠的deepfake检测系统提供了重要见解，并将不确定性量化确定为可信合成媒体检测的基本要求。



## **12. Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2**

消失在稀薄的空气中：针对SAM 2的跨提示通用对抗攻击 cs.CV

Accepted by NeurIPS 2025

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24195v1) [paper-pdf](http://arxiv.org/pdf/2510.24195v1)

**Authors**: Ziqi Zhou, Yifan Hu, Yufei Song, Zijing Li, Shengshan Hu, Leo Yu Zhang, Dezhong Yao, Long Zheng, Hai Jin

**Abstract**: Recent studies reveal the vulnerability of the image segmentation foundation model SAM to adversarial examples. Its successor, SAM2, has attracted significant attention due to its strong generalization capability in video segmentation. However, its robustness remains unexplored, and it is unclear whether existing attacks on SAM can be directly transferred to SAM2. In this paper, we first analyze the performance gap of existing attacks between SAM and SAM2 and highlight two key challenges arising from their architectural differences: directional guidance from the prompt and semantic entanglement across consecutive frames. To address these issues, we propose UAP-SAM2, the first cross-prompt universal adversarial attack against SAM2 driven by dual semantic deviation. For cross-prompt transferability, we begin by designing a target-scanning strategy that divides each frame into k regions, each randomly assigned a prompt, to reduce prompt dependency during optimization. For effectiveness, we design a dual semantic deviation framework that optimizes a UAP by distorting the semantics within the current frame and disrupting the semantic consistency across consecutive frames. Extensive experiments on six datasets across two segmentation tasks demonstrate the effectiveness of the proposed method for SAM2. The comparative results show that UAP-SAM2 significantly outperforms state-of-the-art (SOTA) attacks by a large margin.

摘要: 最近的研究揭示了图像分割基础模型Sam对对抗性示例的脆弱性。其继任者SAM 2因其在视频分割中强大的概括能力而受到广泛关注。然而，其稳健性仍有待探索，目前还不清楚对Sam的现有攻击是否可以直接转移到Sam 2。在本文中，我们首先分析了萨姆和萨姆2之间现有攻击的性能差距，并强调了它们的架构差异带来的两个关键挑战：来自提示的方向引导和连续帧之间的语义纠缠。为了解决这些问题，我们提出了UAP-SAM 2，这是第一个由双重语义偏差驱动的针对SAM 2的交叉提示通用对抗攻击。对于跨提示可移植性，我们首先设计一种目标扫描策略，将每个帧分为k个区域，每个区域随机分配一个提示，以减少优化期间的提示依赖性。为了有效性，我们设计了一个双重语义偏差框架，该框架通过扭曲当前帧内的语义并破坏连续帧之间的语义一致性来优化UAP。在两个分割任务中对六个数据集进行的广泛实验证明了所提出的SAM 2方法的有效性。比较结果表明，UAP-SAM 2明显优于最先进的（SOTA）攻击的大幅度。



## **13. Untargeted Jailbreak Attack**

无目标越狱攻击 cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.02999v2) [paper-pdf](http://arxiv.org/pdf/2510.02999v2)

**Authors**: Xinzhe Huang, Wenjing Hu, Tianhang Zheng, Kedong Xiu, Xiaojun Jia, Di Wang, Zhan Qin, Kui Ren

**Abstract**: Existing gradient-based jailbreak attacks on Large Language Models (LLMs), such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimize adversarial suffixes to align the LLM output with a predefined target response. However, by restricting the optimization objective as inducing a predefined target, these methods inherently constrain the adversarial search space, which limit their overall attack efficacy. Furthermore, existing methods typically require a large number of optimization iterations to fulfill the large gap between the fixed target and the original model response, resulting in low attack efficiency.   To overcome the limitations of targeted jailbreak attacks, we propose the first gradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe response without enforcing any predefined patterns. Specifically, we formulate an untargeted attack objective to maximize the unsafety probability of the LLM response, which can be quantified using a judge model. Since the objective is non-differentiable, we further decompose it into two differentiable sub-objectives for optimizing an optimal harmful response and the corresponding adversarial prompt, with a theoretical analysis to validate the decomposition. In contrast to targeted jailbreak attacks, UJA's unrestricted objective significantly expands the search space, enabling a more flexible and efficient exploration of LLM vulnerabilities.Extensive evaluations demonstrate that UJA can achieve over 80% attack success rates against recent safety-aligned LLMs with only 100 optimization iterations, outperforming the state-of-the-art gradient-based attacks such as I-GCG and COLD-Attack by over 20%.

摘要: 现有的对大型语言模型（LLM）的基于梯度的越狱攻击，例如贪婪协调梯度（GCG）和COLD-Attack，通常会优化对抗性后缀，以将LLM输出与预定义的目标响应保持一致。然而，通过将优化目标限制为诱导预定义的目标，这些方法本质上限制了对抗搜索空间，从而限制了其总体攻击功效。此外，现有方法通常需要大量的优化迭代来满足固定目标和原始模型响应之间的大差距，导致攻击效率低。   为了克服定向越狱攻击的局限性，我们提出了第一个基于梯度的非定向越狱攻击（UJA），旨在在不强制执行任何预定义模式的情况下引发不安全的响应。具体来说，我们制定了一个无针对性的攻击目标，以最大化LLM响应的不安全概率，该概率可以使用判断模型进行量化。由于目标是不可微的，因此我们进一步将其分解为两个可微的子目标，用于优化最佳有害反应和相应的对抗提示，并通过理论分析来验证分解。与有针对性的越狱攻击相比，UJA的无限制目标显着扩大了搜索空间，从而能够更灵活、更高效地探索LLM漏洞。广泛的评估表明，UJA只需100次优化迭代即可针对最近的安全一致LLM实现超过80%的攻击成功率，比I-GCG和COLD-Attack等最先进的基于梯度的攻击性能高出20%以上。



## **14. Learning to Attack: Uncovering Privacy Risks in Sequential Data Releases**

学会攻击：揭露连续数据发布中的隐私风险 cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24807v1) [paper-pdf](http://arxiv.org/pdf/2510.24807v1)

**Authors**: Ziyao Cui, Minxing Zhang, Jian Pei

**Abstract**: Privacy concerns have become increasingly critical in modern AI and data science applications, where sensitive information is collected, analyzed, and shared across diverse domains such as healthcare, finance, and mobility. While prior research has focused on protecting privacy in a single data release, many real-world systems operate under sequential or continuous data publishing, where the same or related data are released over time. Such sequential disclosures introduce new vulnerabilities, as temporal correlations across releases may enable adversaries to infer sensitive information that remains hidden in any individual release. In this paper, we investigate whether an attacker can compromise privacy in sequential data releases by exploiting dependencies between consecutive publications, even when each individual release satisfies standard privacy guarantees. To this end, we propose a novel attack model that captures these sequential dependencies by integrating a Hidden Markov Model with a reinforcement learning-based bi-directional inference mechanism. This enables the attacker to leverage both earlier and later observations in the sequence to infer private information. We instantiate our framework in the context of trajectory data, demonstrating how an adversary can recover sensitive locations from sequential mobility datasets. Extensive experiments on Geolife, Porto Taxi, and SynMob datasets show that our model consistently outperforms baseline approaches that treat each release independently. The results reveal a fundamental privacy risk inherent to sequential data publishing, where individually protected releases can collectively leak sensitive information when analyzed temporally. These findings underscore the need for new privacy-preserving frameworks that explicitly model temporal dependencies, such as time-aware differential privacy or sequential data obfuscation strategies.

摘要: 隐私问题在现代人工智能和数据科学应用中变得越来越重要，敏感信息是在医疗保健、金融和移动等不同领域收集、分析和共享的。虽然之前的研究重点是在单个数据发布中保护隐私，但许多现实世界的系统在顺序或连续数据发布下运行，其中相同或相关的数据会随着时间的推移而发布。此类连续披露引入了新的漏洞，因为版本之间的时间相关性可能使对手能够推断出隐藏在任何单个版本中的敏感信息。在本文中，我们研究攻击者是否可以通过利用连续发布之间的依赖性来损害连续数据发布中的隐私，即使每个单独的发布都满足标准隐私保证。为此，我们提出了一种新颖的攻击模型，该模型通过将隐马尔科夫模型与基于强化学习的双向推理机制集成来捕获这些顺序依赖关系。这使攻击者能够利用序列中早期和后期的观察来推断私人信息。我们在轨迹数据的上下文中实例化了我们的框架，展示了对手如何从顺序移动数据集中恢复敏感位置。对Geolife、Porto Taxi和SynMob数据集的广泛实验表明，我们的模型始终优于独立处理每个版本的基线方法。结果揭示了顺序数据发布固有的基本隐私风险，其中单独受保护的发布在进行临时分析时可能会集体泄露敏感信息。这些发现强调了对新的隐私保护框架的需求，这些框架可以显式地建模时间依赖性，例如时间感知的差异隐私或顺序数据混淆策略。



## **15. Enhancing CLIP Robustness via Cross-Modality Alignment**

通过跨模式对齐增强CLIP稳健性 cs.CV

NeurIPS 2025 Spotlight

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24038v1) [paper-pdf](http://arxiv.org/pdf/2510.24038v1)

**Authors**: Xingyu Zhu, Beier Zhu, Shuo Wang, Kesen Zhao, Hanwang Zhang

**Abstract**: Vision-language models (VLMs) such as CLIP demonstrate strong generalization in zero-shot classification but remain highly vulnerable to adversarial perturbations. Existing methods primarily focus on adversarial fine-tuning or prompt optimization; they often overlook the gaps in CLIP's encoded features, which is shown as the text and image features lie far apart from each other. This misalignment is significantly amplified under adversarial perturbations, leading to severe degradation in classification performance. To address this problem, we propose Cross-modality Alignment, dubbed COLA, an optimal transport-based framework that explicitly addresses adversarial misalignment by restoring both global image-text alignment and local structural consistency in the feature space. (1) COLA first projects adversarial image embeddings onto a subspace spanned by class text features, effectively filtering out non-semantic distortions while preserving discriminative information. (2) It then models images and texts as discrete distributions over multiple augmented views and refines their alignment via OT, with the subspace projection seamlessly integrated into the cost computation. This design ensures stable cross-modal alignment even under adversarial conditions. COLA is training-free and compatible with existing fine-tuned models. Extensive evaluations across 14 zero-shot classification benchmarks demonstrate the effectiveness of COLA, especially with an average improvement of 6.7% on ImageNet and its variants under PGD adversarial attacks, while maintaining high accuracy on clean samples.

摘要: CLIP等视觉语言模型（VLM）在零镜头分类中表现出很强的概括性，但仍然极易受到对抗性扰动的影响。现有的方法主要关注于对抗性微调或即时优化;它们经常忽视CLIP编码特征中的差距，该差距表现为文本和图像特征彼此相距很远。这种不对准在对抗性扰动下被显着放大，导致分类性能严重下降。为了解决这个问题，我们提出了跨模式对齐（称为COLA），这是一个基于传输的最佳框架，通过恢复特征空间中的全局图像-文本对齐和局部结构一致性来明确解决对抗性失调。(1)COLA首先将对抗性图像嵌入投影到由类文本特征跨越的子空间上，有效地过滤掉非语义失真，同时保留区分性信息。(2)然后，它将图像和文本建模为多个增强视图上的离散分布，并通过OT细化它们的对齐，并将子空间投影无缝集成到成本计算中。这种设计即使在对抗条件下也能确保稳定的跨模式对齐。COLA无需培训，并与现有的微调型号兼容。对14个零镜头分类基准的广泛评估证明了COLA的有效性，特别是在PVD对抗攻击下ImageNet及其变体的平均改进了6.7%，同时在干净样本上保持了高准确性。



## **16. A Volumetric Privacy Measure for Dynamical Systems With Bounded Disturbance**

具有有限干扰的动态系统的体积隐私测量 eess.SY

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2501.02893v5) [paper-pdf](http://arxiv.org/pdf/2501.02893v5)

**Authors**: Chuanghong Weng, Ehsan Nekouei

**Abstract**: In this paper, we first present a volumetric privacy measure for dynamical systems with bounded disturbances, wherein the states of the system contain private information and an adversary with access to sensor measurements attempts to infer the set of potential values of the private information. Under the proposed privacy measure, the volume of the uncertainty set of the adversary given the sensor measurements is considered as the privacy level of the system. We next characteristic the time evolution of the proposed privacy measure and study its properties for a particular system with both public and private states, where a set containing the public state is shared as the observation. Approximate set-membership estimation techniques are developed to compute the private-state uncertainty set, and the properties of the privacy measure are analyzed, demonstrating that the uncertainty reduction of the adversary is bounded by the information gain from the observation set. Furthermore, an optimization-based privacy filter design problem is formulated, employing randomization and linear programming to enhance the privacy level. The effectiveness of the proposed approach is validated through a production-inventory case study. Results show that the optimal privacy filter significantly improves robustness against inference attacks and outperforms two baseline mechanisms based on additive noise and quantization.

摘要: 在本文中，我们首先针对具有有界干扰的动态系统提出了一种体积隐私测量，其中系统的状态包含私人信息，并且可以访问传感器测量结果的对手试图推断私人信息的潜在值集。根据提出的隐私测量，给定传感器测量结果的对手不确定性集的量被视为系统的隐私级别。接下来，我们描述了所提出的隐私测量的时间演变，并研究其在具有公共和私人状态的特定系统中的属性，其中包含公共状态的集合作为观察被共享。开发了近似集隶属度估计技术来计算私人状态不确定性集，并分析了隐私度量的性质，证明了对手的不确定性降低受到观察集的信息收益的限制。此外，还提出了一个基于优化的隐私过滤器设计问题，采用随机化和线性规划来提高隐私级别。通过生产库存案例研究验证了所提出方法的有效性。结果表明，最佳隐私过滤器显着提高了对推理攻击的鲁棒性，并且优于基于添加性噪音和量化的两种基线机制。



## **17. Modeling Object Attention in Mobile AR for Intrinsic Cognitive Security**

基于内在认知安全的移动AR中目标注意力建模 cs.HC

Conference Paper, 5 pages. Published at the 2025 ACM the  International Symposium on Theory, Algorithmic Foundations, and Protocol  Design for Mobile Networks and Mobile Computing (MobiHoc)

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24004v1) [paper-pdf](http://arxiv.org/pdf/2510.24004v1)

**Authors**: Shane Dirksen, Radha Kumaran, You-Jin Kim, Yilin Wang, Tobias Höllerer

**Abstract**: We study attention in mobile Augmented Reality (AR) using object recall as a proxy outcome. We observe that the ability to recall an object (physical or virtual) that was encountered in a mobile AR experience depends on many possible impact factors and attributes, with some objects being readily recalled while others are not, and some people recalling objects overall much better or worse than others. This opens up a potential cognitive attack in which adversaries might create conditions that make an AR user not recall certain potentially mission-critical objects. We explore whether a calibrated predictor of object recall can help shield against such cognitive attacks. We pool data from four mobile AR studies (with a total of 1,152 object recall probes) and fit a Partial Least Squares Structural Equation Model (PLS-SEM) with formative Object, Scene, and User State composites predicting recall, also benchmarking against Random Forest and multilayer perceptron classifiers. PLS-SEM attains the best F1 score in three of four studies. Additionally, path estimates identify lighting, augmentation density, AR registration stability, cognitive load, and AR familiarity as primary drivers. The model outputs per-object recall probabilities that can drive interface adjustments when predicted recall falls. Overall, PLS-SEM provides competitive accuracy with interpretable levers for design and evaluation in mobile AR.

摘要: 我们使用对象回忆作为代理结果来研究移动增强现实（AR）中的注意力。我们观察到，回忆起移动AR体验中遇到的对象（物理或虚拟）的能力取决于许多可能的影响因素和属性，有些对象很容易被回忆起，而另一些则不然，有些人回忆起对象总体上比其他人好得多或差得多。这引发了潜在的认知攻击，对手可能会创造条件，使AR用户无法回忆起某些潜在的关键任务对象。我们探讨了对象回忆的校准预测器是否可以帮助抵御此类认知攻击。我们汇集了来自四项移动AR研究的数据（共有1，152个对象召回探针），并拟合了偏最小二乘结构方程模型（PLS-SEM），其中包含预测召回的形成性对象，场景和用户状态复合材料，还对随机森林和多层感知器分类器进行了基准测试。PLS-SEM在四项研究中的三项中获得最佳F1评分。此外，路径估计将照明、增强密度、AR配准稳定性、认知负荷和AR熟悉度确定为主要驱动因素。该模型输出每个对象的召回概率，当预测召回下降时，可以驱动界面调整。总体而言，PLS-TEM为移动AR的设计和评估提供了有竞争力的准确性和可解释的杠杆。



## **18. Fortytwo: Swarm Inference with Peer-Ranked Consensus**

42：具有同行排名共识的群体推理 cs.LG

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.24801v1) [paper-pdf](http://arxiv.org/pdf/2510.24801v1)

**Authors**: Vladyslav Larin, Ihor Naumenko, Aleksei Ivashov, Ivan Nikitin, Alexander Firsov

**Abstract**: As centralized AI hits compute ceilings and diminishing returns from ever-larger training runs, meeting demand requires an inference layer that scales horizontally in both capacity and capability. We present Fortytwo, a novel protocol that leverages swarm intelligence principles and distributed pairwise ranking consensus to achieve superior performance in AI inference. Our approach reimagines collaboration among AI nodes using swarm inference: a peer-ranked, reputation-weighted consensus across heterogeneous models that surfaces the highest-quality responses. Using pairwise ranking with a custom Bradley-Terry-style aggregation model, we demonstrate that swarm inference substantially outperforms majority voting, achieving 85.90% on GPQA Diamond versus 68.69% for majority voting with the same model set - an improvement of +17.21 percentage points (approximately +25.1% relative). The protocol incorporates on-chain reputation so node influence adapts to demonstrated accuracy over time, yielding a meritocratic consensus that filters low-quality or malicious participants. To resist Sybil attacks, Fortytwo employs proof-of-capability in its consensus: nodes must successfully complete calibration/test requests and stake reputation to enter ranking rounds, making multi-identity attacks economically unattractive while preserving openness. Across six challenging benchmarks, including GPQA Diamond, LiveCodeBench, and AIME, our evaluation indicates higher accuracy and strong resilience to adversarial and noisy free-form prompting (e.g., prompt-injection degradation of only 0.12% versus 6.20% for a monolithic single-model baseline), while retaining practical deployability. Together, these results establish a foundation for decentralized AI systems - democratizing access to high-quality inference through collective intelligence without sacrificing reliability or security.

摘要: 随着集中式人工智能达到计算上限，并且越来越大规模的培训运行带来的回报不断减少，满足需求需要一个在容量和能力方面水平扩展的推理层。我们介绍了Fortytwo，这是一种新型协议，利用群体智能原则和分布式成对排名共识来实现人工智能推理的卓越性能。我们的方法使用群体推理重新构想人工智能节点之间的协作：跨异类模型的同行排名、声誉加权共识，以呈现最高质量的响应。使用自定义Bradley-Terry式聚合模型的成对排名，我们证明群体推理的表现大大优于多数投票，GPQA Diamond上的比例为85.90%，而使用相同模型集的多数投票为68.69%-提高了+17.21个百分点（相对约+25.1%）。该协议结合了链上声誉，以便随着时间的推移，节点影响力适应所证明的准确性，从而产生精英共识，过滤低质量或恶意参与者。为了抵抗Sybil攻击，Fortytwo在其共识中采用了能力证明：节点必须成功完成校准/测试请求并赢得声誉才能进入排名轮，这使得多身份攻击在经济上没有吸引力，同时保持开放性。在六个具有挑战性的基准测试中，包括GPQA Diamond、LiveCodeBench和AIME，我们的评估表明对对抗性和有噪音的自由形式提示具有更高的准确性和较强的弹性（例如，预算注入降级仅为0.12%，而整体单一型号基线为6.20%），同时保留了实际的可部署性。这些结果共同为去中心化人工智能系统奠定了基础--在不牺牲可靠性或安全性的情况下，通过集体智能实现高质量推理的民主化。



## **19. Secure Control of Connected and Autonomous Electrified Vehicles Under Adversarial Cyber-Attacks**

对抗性网络攻击下互联和自主电动汽车的安全控制 eess.SY

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.23922v1) [paper-pdf](http://arxiv.org/pdf/2510.23922v1)

**Authors**: Shashank Dhananjay Vyas, Satadru Dey

**Abstract**: Connected and Autonomous Electrified Vehicles (CAEV) is the solution to the future smart mobility having benefits of efficient traffic flow and cleaner environmental impact. Although CAEV has advantages they are still susceptible to adversarial cyber attacks due to their autonomous electric operation and the involved connectivity. To alleviate this issue, we propose a secure control architecture of CAEV. Particularly, we design an additional control input using Reinforcement Learning (RL) to be applied to the vehicle powertrain along with the input commanded by the battery. We present simulation case studies to demonstrate the potential of the proposed approach in keeping the CAEV platoon operating safely without collisions by curbing the effect of adversarial attacks.

摘要: 互联和自主电动汽车（CAEV）是未来智能出行的解决方案，具有高效的交通流量和更清洁的环境影响。尽管CAEV具有优势，但由于其自主电动操作和相关的连接性，它们仍然容易受到对抗性网络攻击。为了缓解这个问题，我们提出了CAEV的安全控制架构。特别是，我们使用强化学习（RL）设计了额外的控制输入，与电池命令的输入一起应用于车辆动力总成。我们提出了模拟案例研究，以证明所提出的方法通过遏制对抗攻击的影响来保持CAEV排安全运行而无碰撞的潜力。



## **20. Securing Transfer-Learned Networks with Reverse Homomorphic Encryption**

使用反向同形加密保护传输学习网络 cs.CR

added protection via RHE and black box attacks

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2505.14323v2) [paper-pdf](http://arxiv.org/pdf/2505.14323v2)

**Authors**: Robert Allison, Tomasz Maciążek, Henry Bourne

**Abstract**: The growing body of literature on training-data reconstruction attacks raises significant concerns about deploying neural network classifiers trained on sensitive data. However, differentially private (DP) training (e.g. using DP-SGD) can defend against such attacks with large training datasets causing only minimal loss of network utility. Folklore, heuristics, and (albeit pessimistic) DP bounds suggest this fails for networks trained with small per-class datasets, yet to the best of our knowledge the literature offers no compelling evidence. We directly demonstrate this vulnerability by significantly extending reconstruction attack capabilities under a realistic adversary threat model for few-shot transfer learned image classifiers. We design new white-box and black-box attacks and find that DP-SGD is unable to defend against these without significant classifier utility loss. To address this, we propose a novel homomorphic encryption (HE) method that protects training data without degrading model's accuracy. Conventional HE secures model's input data and requires costly homomorphic implementation of the entire classifier. In contrast, our new scheme is computationally efficient and protects training data rather than input data. This is achieved by means of a simple role-reversal where classifier input data is unencrypted but transfer-learned weights are encrypted. Classifier outputs remain encrypted, thus preventing both white-box and black-box (and any other) training-data reconstruction attacks. Under this new scheme only a trusted party with a private decryption key can obtain the classifier class decisions.

摘要: 越来越多的关于训练数据重建攻击的文献引发了人们对部署在敏感数据上训练的神经网络分类器的严重担忧。然而，差异私密（DP）训练（例如使用DP-BCD）可以通过大型训练数据集抵御此类攻击，只会导致最小的网络效用损失。民间传说、启发式和（尽管悲观）DP界限表明，对于使用每类较小数据集训练的网络来说，这是失败的，但据我们所知，文献没有提供令人信服的证据。我们通过在现实的对手威胁模型下显着扩展重建攻击能力来直接证明这一漏洞。我们设计了新的白盒和黑匣子攻击，发现DP-BCD无法在不造成重大分类器效用损失的情况下抵御这些攻击。为了解决这个问题，我们提出了一种新型的同质加密（HE）方法，该方法可以保护训练数据，而不会降低模型的准确性。传统的HE保护模型的输入数据，并且需要整个分类器的昂贵的homomorphic实现。相比之下，我们的新方案计算效率高，并且保护训练数据而不是输入数据。这是通过简单的角色倒置来实现的，其中分类器输入数据未加密，但转移学习权重被加密。分类器输出保持加密，从而防止白盒和黑匣子（以及任何其他）训练数据重建攻击。在这个新方案下，只有具有私有解密密钥的可信方才能获得分类器类别决策。



## **21. Apollo: A Posteriori Label-Only Membership Inference Attack Towards Machine Unlearning**

Apollo：针对机器取消学习的后验纯标签成员推理攻击 cs.LG

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2506.09923v2) [paper-pdf](http://arxiv.org/pdf/2506.09923v2)

**Authors**: Liou Tang, James Joshi, Ashish Kundu

**Abstract**: Machine Unlearning (MU) aims to update Machine Learning (ML) models following requests to remove training samples and their influences on a trained model efficiently without retraining the original ML model from scratch. While MU itself has been employed to provide privacy protection and regulatory compliance, it can also increase the attack surface of the model. Existing privacy inference attacks towards MU that aim to infer properties of the unlearned set rely on the weaker threat model that assumes the attacker has access to both the unlearned model and the original model, limiting their feasibility toward real-life scenarios. We propose a novel privacy attack, A Posteriori Label-Only Membership Inference Attack towards MU, Apollo, that infers whether a data sample has been unlearned, following a strict threat model where an adversary has access to the label-output of the unlearned model only. We demonstrate that our proposed attack, while requiring less access to the target model compared to previous attacks, can achieve relatively high precision on the membership status of the unlearned samples.

摘要: 机器非学习（MU）旨在根据请求更新机器学习（ML）模型，以有效地删除训练样本及其对训练模型的影响，而无需从头开始重新训练原始ML模型。虽然MU本身被用来提供隐私保护和法规遵从性，但它也可以增加模型的攻击面。针对MU的现有隐私推断攻击旨在推断未学习集的属性，依赖于较弱的威胁模型，该模型假设攻击者可以访问未学习模型和原始模型，从而限制了其对现实生活场景的可行性。我们提出了一种新的隐私攻击，一个后验标签只有成员资格推理攻击对MU，阿波罗，推断数据样本是否已被unlearned，以下严格的威胁模型，对手只能访问未学习模型的标签输出。我们证明，与之前的攻击相比，我们提出的攻击虽然需要更少的访问目标模型，但可以对未学习样本的成员身份状态实现相对高的精确度。



## **22. UNDREAM: Bridging Differentiable Rendering and Photorealistic Simulation for End-to-end Adversarial Attacks**

UNDREAM：为端到端对抗性攻击搭建差异渲染和真实感模拟的桥梁 cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.16923v2) [paper-pdf](http://arxiv.org/pdf/2510.16923v2)

**Authors**: Mansi Phute, Matthew Hull, Haoran Wang, Alec Helbling, ShengYun Peng, Willian Lunardi, Martin Andreoni, Wenke Lee, Duen Horng Chau

**Abstract**: Deep learning models deployed in safety critical applications like autonomous driving use simulations to test their robustness against adversarial attacks in realistic conditions. However, these simulations are non-differentiable, forcing researchers to create attacks that do not integrate simulation environmental factors, reducing attack success. To address this limitation, we introduce UNDREAM, the first software framework that bridges the gap between photorealistic simulators and differentiable renderers to enable end-to-end optimization of adversarial perturbations on any 3D objects. UNDREAM enables manipulation of the environment by offering complete control over weather, lighting, backgrounds, camera angles, trajectories, and realistic human and object movements, thereby allowing the creation of diverse scenes. We showcase a wide array of distinct physically plausible adversarial objects that UNDREAM enables researchers to swiftly explore in different configurable environments. This combination of photorealistic simulation and differentiable optimization opens new avenues for advancing research of physical adversarial attacks.

摘要: 部署在自动驾驶等安全关键应用中的深度学习模型使用模拟来测试其在现实条件下对抗攻击的稳健性。然而，这些模拟是不可区分的，迫使研究人员创建不集成模拟环境因素的攻击，从而降低了攻击成功率。为了解决这一局限性，我们引入了UNDREAM，这是第一个弥合真实感模拟器和可区分渲染器之间差距的软件框架，能够对任何3D对象上的对抗性扰动进行端到端优化。UNDREAM通过对天气、灯光、背景、摄像机角度、轨迹以及真实的人和物体运动进行完全控制来操纵环境，从而允许创建多样化的场景。我们展示了各种不同的物理上看似合理的对抗对象，UNDREAM使研究人员能够在不同的可配置环境中快速探索。真实感模拟和可区分优化的结合为推进物理对抗攻击的研究开辟了新的途径。



## **23. On the Stability of Graph Convolutional Neural Networks: A Probabilistic Perspective**

图卷积神经网络的稳定性：概率的角度 cs.LG

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2506.01213v4) [paper-pdf](http://arxiv.org/pdf/2506.01213v4)

**Authors**: Ning Zhang, Henry Kenlay, Li Zhang, Mihai Cucuringu, Xiaowen Dong

**Abstract**: Graph convolutional neural networks (GCNNs) have emerged as powerful tools for analyzing graph-structured data, achieving remarkable success across diverse applications. However, the theoretical understanding of the stability of these models, i.e., their sensitivity to small changes in the graph structure, remains in rather limited settings, hampering the development and deployment of robust and trustworthy models in practice. To fill this gap, we study how perturbations in the graph topology affect GCNN outputs and propose a novel formulation for analyzing model stability. Unlike prior studies that focus only on worst-case perturbations, our distribution-aware formulation characterizes output perturbations across a broad range of input data. This way, our framework enables, for the first time, a probabilistic perspective on the interplay between the statistical properties of the node data and perturbations in the graph topology. We conduct extensive experiments to validate our theoretical findings and demonstrate their benefits over existing baselines, in terms of both representation stability and adversarial attacks on downstream tasks. Our results demonstrate the practical significance of the proposed formulation and highlight the importance of incorporating data distribution into stability analysis.

摘要: 图卷积神经网络（GCNN）已成为分析图结构数据的强大工具，在各种应用中取得了显着的成功。然而，对这些模型稳定性的理论理解，即，它们对图形结构中的微小变化的敏感性仍然在相当有限的环境中，阻碍了在实践中稳健且值得信赖的模型的开发和部署。为了填补这一空白，我们研究了图布局中的扰动如何影响GCNN输出，并提出了一种用于分析模型稳定性的新公式。与之前仅关注最坏情况扰动的研究不同，我们的分布感知公式描述了广泛输入数据中的输出扰动。通过这种方式，我们的框架首次能够从概率角度看待节点数据的统计属性和图布局中的扰动之间的相互作用。我们进行了广泛的实验来验证我们的理论发现，并展示它们在表示稳定性和对下游任务的对抗攻击方面相对于现有基线的优势。我们的结果证明了拟议公式的实际意义，并强调了将数据分布纳入稳定性分析的重要性。



## **24. Authentication Against Insecure Bootstrapping for 5G Networks: Feasibility, Resiliency, and Transitional Solutions in Post-Quantum Era**

针对5G网络不安全引导的认证：后量子时代的可行性、弹性和过渡解决方案 cs.CR

17 pages, 3 tables, 6 figures

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.23457v1) [paper-pdf](http://arxiv.org/pdf/2510.23457v1)

**Authors**: Saleh Darzi, Mirza Masfiqur Rahman, Imtiaz Karim, Rouzbeh Behnia, Attila A Yavuz, Elisa Bertino

**Abstract**: The 5G protocol lacks a robust base station authentication mechanism during the initial bootstrapping phase, leaving it susceptible to threats such as fake base station attacks. Conventional solutions, including digital signatures based on Public Key Infrastructures (PKIs) and identity-based signatures, are inadequate against quantum-capable adversaries. While integrating NIST's Post-Quantum Cryptography (PQC) standards is a leading approach for quantum resistance, their suitability for 5G base station authentication remains unexplored. Moreover, current solutions are predominantly centralized and lack security features such as distributed authentication. This work presents, to our knowledge, the first comprehensive network-level performance characterization of integrating NIST-PQC standards and conventional digital signatures (including threshold and identity-based schemes) into 5G base station authentication. Our findings reveal significant feasibility concerns, with direct PQC adoption hindered by protocol constraints and large signature sizes. We also highlight the performance limitations of conventional methods due to the overhead of certificate chains. To mitigate these challenges, we propose BORG, a transitional authentication solution based on a Hierarchical Identity-Based Threshold Signature scheme with a Fail-Stop property. BORG offers post-mortem post-quantum forgery detection and distributed trust via threshold and compact signatures, well-suited for 5G's stringent requirements. Our performance analysis underscores an important warning on the infeasibility of direct PQC integration and positions BORG as an effective transitional solution toward future quantum-resilient 5G authentication.

摘要: 5G协议在初始引导阶段缺乏强大的基站认证机制，容易受到假基站攻击等威胁。传统的解决方案，包括基于公钥结构（PKI）的数字签名和基于身份的签名，不足以对抗具有量子能力的对手。虽然集成NIH的后量子密码学（PQC）标准是量子抵抗的领先方法，但其对5G基站认证的适用性仍有待探索。此外，当前的解决方案主要是集中式的，缺乏分布式身份验证等安全功能。据我们所知，这项工作提出了将NIST-PQC标准和传统数字签名（包括阈值和基于身份的方案）集成到5G基站认证中的第一个全面的网络级性能表征。我们的研究结果揭示了重大的可行性问题，直接PQC采用协议约束和大签名大小的阻碍。我们还强调了传统方法由于证书链的开销而导致的性能限制。为了缓解这些挑战，我们提出了BORG，一个过渡的认证解决方案的基础上的分层的基于身份的门限签名方案的失败停止属性。BORG通过阈值和紧凑签名提供事后量子伪造检测和分布式信任，非常适合5G的严格要求。我们的性能分析强调了直接PQC集成不可行的重要警告，并将BORG定位为未来量子弹性5G认证的有效过渡解决方案。



## **25. Generalization Bounds for Robust Contrastive Learning: From Theory to Practice**

稳健对比学习的概括界限：从理论到实践 cs.LG

13 pages, 1 figure, 4 tables

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2311.09671v2) [paper-pdf](http://arxiv.org/pdf/2311.09671v2)

**Authors**: Ngoc N. Tran, Lam Tran, Hoang Phan, Anh Bui, Tung Pham, Toan Tran, Dinh Phung, Trung Le

**Abstract**: Contrastive Learning first extracts features from unlabeled data, followed by linear probing with labeled data. Adversarial Contrastive Learning (ACL) integrates Adversarial Training into the first phase to enhance feature robustness against attacks in the probing phase. While ACL has shown strong empirical results, its theoretical understanding remains limited. Furthermore, while a fair amount of theoretical works analyze how the unsupervised loss can support the supervised loss in the probing phase, none has examined its role to the robust supervised loss. To fill this gap, our work develops rigorous theories to identify which components in the unsupervised training can help improve the robust supervised loss. Specifically, besides the adversarial contrastive loss, we reveal that the benign one, along with a global divergence between benign and adversarial examples can also improve robustness. Proper experiments are conducted to justify our findings.

摘要: 对比学习首先从未标记的数据中提取特征，然后使用标记的数据进行线性探测。对抗性对比学习（ACL）将对抗性训练集成到第一阶段，以增强特征针对探测阶段攻击的稳健性。虽然ACL表现出了强有力的实证结果，但其理论理解仍然有限。此外，虽然大量理论著作分析了无监督损失如何在探测阶段支持监督损失，但没有人研究过它对稳健监督损失的作用。为了填补这一空白，我们的工作开发了严格的理论来确定无监督训练中的哪些组件可以帮助改善稳健的监督损失。具体来说，除了对抗性对比损失之外，我们还发现良性对比损失以及良性和对抗性示例之间的全球差异也可以提高稳健性。进行适当的实验来证明我们的发现的合理性。



## **26. Attention! Your Vision Language Model Could Be Maliciously Manipulated**

注意！您的视觉语言模型可能被恶意操纵 cs.CV

NeurIPS 2025

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2505.19911v2) [paper-pdf](http://arxiv.org/pdf/2505.19911v2)

**Authors**: Xiaosen Wang, Shaokang Wang, Zhijin Ge, Yuyang Luo, Shudong Zhang

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets. Code is available at https://github.com/Trustworthy-AI-Group/VMA.

摘要: 大型视觉语言模型（VLM）在理解复杂的现实世界场景和支持数据驱动的决策流程方面取得了显着的成功。然而，VLM对对抗性示例（无论是文本还是图像）表现出显着的脆弱性，这可能会导致各种对抗性结果，例如越狱、劫持和幻觉等。在这项工作中，我们从经验和理论上证明了VLM特别容易受到基于图像的对抗示例的影响，其中不可感知的扰动可以精确地操纵每个输出令牌。为此，我们提出了一种名为视觉语言模型操纵攻击（VMA）的新型攻击，该攻击将一阶和二阶动量优化技术与可微转换机制集成在一起，以有效地优化对抗性扰动。值得注意的是，VMA可以是一把双刃剑：它可以被用来实施各种攻击，例如越狱、劫持、隐私泄露、拒绝服务和海绵示例的生成等，同时允许注入水印以进行版权保护。广泛的实证评估证实了VMA在不同场景和数据集中的有效性和普遍性。代码可在https://github.com/Trustworthy-AI-Group/VMA上获取。



## **27. Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction**

通过减少教学不确定性探索语义约束对抗示例 cs.AI

NeurIPS 2025

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.22981v1) [paper-pdf](http://arxiv.org/pdf/2510.22981v1)

**Authors**: Jin Hu, Jiakai Wang, Linna Jing, Haolin Li, Haodong Liu, Haotong Qin, Aishan Liu, Ke Xu, Xianglong Liu

**Abstract**: Recently, semantically constrained adversarial examples (SemanticAE), which are directly generated from natural language instructions, have become a promising avenue for future research due to their flexible attacking forms. To generate SemanticAEs, current methods fall short of satisfactory attacking ability as the key underlying factors of semantic uncertainty in human instructions, such as referring diversity, descriptive incompleteness, and boundary ambiguity, have not been fully investigated. To tackle the issues, this paper develops a multi-dimensional instruction uncertainty reduction (InSUR) framework to generate more satisfactory SemanticAE, i.e., transferable, adaptive, and effective. Specifically, in the dimension of the sampling method, we propose the residual-driven attacking direction stabilization to alleviate the unstable adversarial optimization caused by the diversity of language references. By coarsely predicting the language-guided sampling process, the optimization process will be stabilized by the designed ResAdv-DDIM sampler, therefore releasing the transferable and robust adversarial capability of multi-step diffusion models. In task modeling, we propose the context-encoded attacking scenario constraint to supplement the missing knowledge from incomplete human instructions. Guidance masking and renderer integration are proposed to regulate the constraints of 2D/3D SemanticAE, activating stronger scenario-adapted attacks. Moreover, in the dimension of generator evaluation, we propose the semantic-abstracted attacking evaluation enhancement by clarifying the evaluation boundary, facilitating the development of more effective SemanticAE generators. Extensive experiments demonstrate the superiority of the transfer attack performance of InSUR. Moreover, we realize the reference-free generation of semantically constrained 3D adversarial examples for the first time.

摘要: 近年来，直接从自然语言指令生成的语义约束对抗示例（SemanticAE）因其灵活的攻击形式而成为未来研究的一个有希望的途径。为了生成SemanticAE，当前的方法缺乏令人满意的攻击能力，因为人类指令中语义不确定性的关键潜在因素，例如引用多样性、描述性不完整性和边界模糊性，尚未得到充分研究。为了解决这些问题，本文开发了一个多维指令不确定性减少（Insus）框架，以生成更满意的SemanticAE，即可转移、适应性和有效。具体来说，在抽样方法维度上，我们提出了剩余驱动的攻击方向稳定化，以缓解语言引用多样性造成的不稳定对抗优化。通过粗略预测语言引导的采样过程，优化过程将由设计的ResAdv-DDIM采样器稳定，从而释放多步扩散模型的可转移且鲁棒的对抗能力。在任务建模中，我们提出了上下文编码的攻击场景约束来补充不完整人类指令中缺失的知识。提出引导掩蔽和渲染器集成来调节2D/3D SemanticAE的约束，从而激活更强的自适应攻击。此外，在生成器评估维度上，我们通过明确评估边界提出了语义抽象的攻击评估增强，促进了更有效的SemanticAE生成器的开发。大量实验证明了Insus传输攻击性能的优越性。此外，我们首次实现了语义约束的3D对抗示例的无引用生成。



## **28. PersonaTeaming: Exploring How Introducing Personas Can Improve Automated AI Red-Teaming**

角色协作：探索引入角色协作如何改善自动化人工智能红色协作 cs.AI

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2509.03728v3) [paper-pdf](http://arxiv.org/pdf/2509.03728v3)

**Authors**: Wesley Hanwen Deng, Sunnie S. Y. Kim, Akshita Jha, Ken Holstein, Motahhare Eslami, Lauren Wilcox, Leon A Gatys

**Abstract**: Recent developments in AI governance and safety research have called for red-teaming methods that can effectively surface potential risks posed by AI models. Many of these calls have emphasized how the identities and backgrounds of red-teamers can shape their red-teaming strategies, and thus the kinds of risks they are likely to uncover. While automated red-teaming approaches promise to complement human red-teaming by enabling larger-scale exploration of model behavior, current approaches do not consider the role of identity. As an initial step towards incorporating people's background and identities in automated red-teaming, we develop and evaluate a novel method, PersonaTeaming, that introduces personas in the adversarial prompt generation process to explore a wider spectrum of adversarial strategies. In particular, we first introduce a methodology for mutating prompts based on either "red-teaming expert" personas or "regular AI user" personas. We then develop a dynamic persona-generating algorithm that automatically generates various persona types adaptive to different seed prompts. In addition, we develop a set of new metrics to explicitly measure the "mutation distance" to complement existing diversity measurements of adversarial prompts. Our experiments show promising improvements (up to 144.1%) in the attack success rates of adversarial prompts through persona mutation, while maintaining prompt diversity, compared to RainbowPlus, a state-of-the-art automated red-teaming method. We discuss the strengths and limitations of different persona types and mutation methods, shedding light on future opportunities to explore complementarities between automated and human red-teaming approaches.

摘要: 人工智能治理和安全研究的最新进展呼吁采取红色团队方法，以有效地揭示人工智能模型带来的潜在风险。其中许多电话都强调了红色团队成员的身份和背景如何影响他们的红色团队策略，从而也强调了他们可能发现的风险。虽然自动化红色团队方法有望通过更大规模地探索模型行为来补充人类红色团队，但目前的方法没有考虑身份的作用。作为将人们的背景和身份融入自动化红色团队的第一步，我们开发并评估了一种新型方法PersonaTeaming，该方法在对抗性提示生成过程中引入角色，以探索更广泛的对抗策略。特别是，我们首先引入了一种基于“红色团队专家”角色或“普通人工智能用户”角色来变异提示的方法。然后，我们开发了一个动态角色生成算法，该算法自动生成适应不同种子提示的各种角色类型。此外，我们开发了一组新的指标来明确测量“突变距离”，以补充现有的对抗提示多样性测量。我们的实验显示，与最先进的自动化红色团队方法RainbowPlus相比，通过角色突变的对抗提示的攻击成功率有了有希望的改进（高达144.1%），同时保持了提示的多样性。我们讨论了不同角色类型和突变方法的优点和局限性，揭示了未来探索自动化和人类红色团队方法之间互补性的机会。



## **29. CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents**

CompressionAttack：利用即时压缩作为LLM支持的代理中的新攻击表面 cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.22963v1) [paper-pdf](http://arxiv.org/pdf/2510.22963v1)

**Authors**: Zesen Liu, Zhixiang Zhang, Yuchong Xie, Dongdong She

**Abstract**: LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to 80% attack success and 98% preference flips, while remaining highly stealthy and transferable. Case studies in VSCode Cline and Ollama confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.

摘要: LLM支持的代理通常使用即时压缩来降低推理成本，但这会带来新的安全风险。压缩模块针对效率而不是安全性进行了优化，可以通过对抗输入来操纵，从而导致语义漂移并改变LLM行为。这项工作将即时压缩确定为一种新型攻击表面，并提出了第一个利用它的框架CompressionAttack。CompressionAttack包括两种策略：HardCom，使用离散对抗编辑进行硬压缩，以及SoftCom，为软压缩执行潜伏空间扰动。对多个LLM的实验显示，攻击成功率高达80%，偏好翻转率高达98%，同时保持高度隐蔽性和可转移性。VSCode Cline和Olama的案例研究证实了现实世界的影响，而当前的防御措施被证明无效，凸显了加强保护的必要性。



## **30. Your Compiler is Backdooring Your Model: Understanding and Exploiting Compilation Inconsistency Vulnerabilities in Deep Learning Compilers**

您的编译器正在为您的模型做后门：了解和利用深度学习编译器中的编译不一致漏洞 cs.CR

This paper is accepted to IEEE S&P 2026, the code is available at  https://github.com/SeekingDream/DLCompilerAttack

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2509.11173v3) [paper-pdf](http://arxiv.org/pdf/2509.11173v3)

**Authors**: Simin Chen, Jinjun Peng, Yixin He, Junfeng Yang, Baishakhi Ray

**Abstract**: Deep learning (DL) compilers are core infrastructure in modern DL systems, offering flexibility and scalability beyond vendor-specific libraries. This work uncovers a fundamental vulnerability in their design: can an official, unmodified compiler alter a model's semantics during compilation and introduce hidden backdoors? We study both adversarial and natural settings. In the adversarial case, we craft benign models where triggers have no effect pre-compilation but become effective backdoors after compilation. Tested on six models, three commercial compilers, and two hardware platforms, our attack yields 100% success on triggered inputs while preserving normal accuracy and remaining undetected by state-of-the-art detectors. The attack generalizes across compilers, hardware, and floating-point settings. In the natural setting, we analyze the top 100 HuggingFace models (including one with 220M+ downloads) and find natural triggers in 31 models. This shows that compilers can introduce risks even without adversarial manipulation.   Our results reveal an overlooked threat: unmodified DL compilers can silently alter model semantics. To our knowledge, this is the first work to expose inherent security risks in DL compiler design, opening a new direction for secure and trustworthy ML.

摘要: 深度学习（DL）编译器是现代DL系统的核心基础设施，提供超出供应商特定库的灵活性和可扩展性。这项工作揭示了他们设计中的一个根本漏洞：官方的、未经修改的编译器能否在编译期间改变模型的语义并引入隐藏的后门？我们研究对抗环境和自然环境。在对抗性的情况下，我们构建了良性模型，其中触发器对预编译没有影响，但在编译后成为有效的后门。经过六种型号、三种商业编译器和两个硬件平台的测试，我们的攻击在触发的输入上取得了100%的成功，同时保持正常的准确性并保持未被最先进的检测器检测到。这种攻击可以跨编译器、硬件和浮点设置进行推广。在自然环境中，我们分析了排名前100的HuggingFace模型（包括下载量超过2.2亿的模型），并在31个模型中找到自然触发因素。这表明，即使没有对抗性操纵，编译器也会引入风险。   我们的结果揭示了一个被忽视的威胁：未修改的DL编译器可以悄悄改变模型语义。据我们所知，这是第一个暴露DL编译器设计中固有安全风险的工作，为安全和可信的ML开辟了新的方向。



## **31. QuantumShield: Multilayer Fortification for Quantum Federated Learning**

QuantumShield：量子联邦学习的多层防御 cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.22945v1) [paper-pdf](http://arxiv.org/pdf/2510.22945v1)

**Authors**: Dev Gurung, Shiva Raj Pokhrel

**Abstract**: In this paper, we propose a groundbreaking quantum-secure federated learning (QFL) framework designed to safeguard distributed learning systems against the emerging threat of quantum-enabled adversaries. As classical cryptographic methods become increasingly vulnerable to quantum attacks, our framework establishes a resilient security architecture that remains robust even in the presence of quantum-capable attackers. We integrate and rigorously evaluate advanced quantum and post-quantum protocols including Quantum Key Distribution (QKD), Quantum Teleportation, Key Encapsulation Mechanisms (KEM) and Post-Quantum Cryptography (PQC) to fortify the QFL process against both classical and quantum threats. These mechanisms are systematically analyzed and implemented to demonstrate their seamless interoperability within a secure and scalable QFL ecosystem. Through comprehensive theoretical modeling and experimental validation, this work provides a detailed security and performance assessment of the proposed framework. Our findings lay a strong foundation for next-generation federated learning systems that are inherently secure in the quantum era.

摘要: 在本文中，我们提出了一个开创性的量子安全联邦学习（QFL）框架，旨在保护分布式学习系统免受量子对手的新威胁。随着经典加密方法变得越来越容易受到量子攻击，我们的框架建立了一个有弹性的安全架构，即使存在具有量子能力的攻击者，该架构也保持稳健。我们集成并严格评估先进的量子和后量子协议，包括量子密钥分发（QKD）、量子隐形传输、密钥封装机制（KEM）和后量子密码学（PQC），以加强QFL流程，抵御经典和量子威胁。这些机制经过系统性分析和实施，以展示它们在安全和可扩展的QFL生态系统中的无缝互操作性。通过全面的理论建模和实验验证，这项工作提供了对拟议框架的详细安全性和性能评估。我们的研究结果为量子时代固有安全的下一代联邦学习系统奠定了坚实的基础。



## **32. Is Your Prompt Poisoning Code? Defect Induction Rates and Security Mitigation Strategies**

您的提示中毒代码是吗？缺陷诱导率和安全缓解策略 cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.22944v1) [paper-pdf](http://arxiv.org/pdf/2510.22944v1)

**Authors**: Bin Wang, YiLu Zhong, MiDi Wan, WenJie Yu, YuanBing Ouyang, Yenan Huang, Hui Li

**Abstract**: Large language models (LLMs) have become indispensable for automated code generation, yet the quality and security of their outputs remain a critical concern. Existing studies predominantly concentrate on adversarial attacks or inherent flaws within the models. However, a more prevalent yet underexplored issue concerns how the quality of a benign but poorly formulated prompt affects the security of the generated code. To investigate this, we first propose an evaluation framework for prompt quality encompassing three key dimensions: goal clarity, information completeness, and logical consistency. Based on this framework, we construct and publicly release CWE-BENCH-PYTHON, a large-scale benchmark dataset containing tasks with prompts categorized into four distinct levels of normativity (L0-L3). Extensive experiments on multiple state-of-the-art LLMs reveal a clear correlation: as prompt normativity decreases, the likelihood of generating insecure code consistently and markedly increases. Furthermore, we demonstrate that advanced prompting techniques, such as Chain-of-Thought and Self-Correction, effectively mitigate the security risks introduced by low-quality prompts, substantially improving code safety. Our findings highlight that enhancing the quality of user prompts constitutes a critical and effective strategy for strengthening the security of AI-generated code.

摘要: 大型语言模型（LLM）对于自动代码生成来说已不可或缺，但其输出的质量和安全性仍然是一个关键问题。现有的研究主要集中在对抗攻击或模型内的固有缺陷上。然而，一个更普遍但未充分研究的问题涉及良性但制定不当的提示的质量如何影响生成代码的安全性。为了研究这一点，我们首先提出了一个包含三个关键维度的即时质量评估框架：目标清晰度、信息完整性和逻辑一致性。基于此框架，我们构建并公开发布CWE-BENCH-PYTHON，这是一个大规模基准数据集，包含将提示分为四个不同的规范性级别（L0-L3）的任务。对多个最先进的LLM进行的广泛实验揭示了明显的相关性：随着即时规范性的降低，生成不安全代码的可能性持续且显着增加。此外，我们还证明，思想链和自我纠正等先进的提示技术可以有效地减轻低质量提示带来的安全风险，大大提高代码安全性。我们的研究结果强调，提高用户提示的质量是加强人工智能生成代码安全性的关键而有效的策略。



## **33. Self-Calibrated Consistency can Fight Back for Adversarial Robustness in Vision-Language Models**

自校准一致性可以反击视觉语言模型中的对抗鲁棒性 cs.CV

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22785v1) [paper-pdf](http://arxiv.org/pdf/2510.22785v1)

**Authors**: Jiaxiang Liu, Jiawei Du, Xiao Liu, Prayag Tiwari, Mingkun Xu

**Abstract**: Pre-trained vision-language models (VLMs) such as CLIP have demonstrated strong zero-shot capabilities across diverse domains, yet remain highly vulnerable to adversarial perturbations that disrupt image-text alignment and compromise reliability. Existing defenses typically rely on adversarial fine-tuning with labeled data, limiting their applicability in zero-shot settings. In this work, we identify two key weaknesses of current CLIP adversarial attacks -- lack of semantic guidance and vulnerability to view variations -- collectively termed semantic and viewpoint fragility. To address these challenges, we propose Self-Calibrated Consistency (SCC), an effective test-time defense. SCC consists of two complementary modules: Semantic consistency, which leverages soft pseudo-labels from counterattack warm-up and multi-view predictions to regularize cross-modal alignment and separate the target embedding from confusable negatives; and Spatial consistency, aligning perturbed visual predictions via augmented views to stabilize inference under adversarial perturbations. Together, these modules form a plug-and-play inference strategy. Extensive experiments on 22 benchmarks under diverse attack settings show that SCC consistently improves the zero-shot robustness of CLIP while maintaining accuracy, and can be seamlessly integrated with other VLMs for further gains. These findings highlight the great potential of establishing an adversarially robust paradigm from CLIP, with implications extending to broader vision-language domains such as BioMedCLIP.

摘要: CLIP等预训练的视觉语言模型（VLM）已在不同领域表现出强大的零攻击能力，但仍然极易受到破坏图像与文本对齐并损害可靠性的对抗性扰动的影响。现有的防御通常依赖于对标记数据的对抗微调，限制了其在零射击设置中的适用性。在这项工作中，我们确定了当前CLIP对抗攻击的两个关键弱点--缺乏语义指导和查看变化的脆弱性--统称为语义脆弱性和观点脆弱性。为了应对这些挑战，我们提出了自校准一致性（SCC），这是一种有效的测试时防御措施。SCC由两个补充模块组成：语义一致性，利用来自反击热身和多视图预测的软伪标签来规范跨模式对齐，并将目标嵌入与可混淆的阴性区分开来;空间一致性，通过增强视图对齐受干扰的视觉预测，以稳定对抗性扰动下的推理。这些模块共同构成了即插即用推理策略。在不同攻击设置下对22个基准进行的大量实验表明，SCC在保持准确性的同时持续提高了CLIP的零触发鲁棒性，并且可以与其他VLM无缝集成以获得进一步的收益。这些发现凸显了从CLIP中建立对抗强大范式的巨大潜力，其影响延伸到更广泛的视觉语言领域，例如BioMedCLIP。



## **34. SpoofTrackBench: Interpretable AI for Spoof-Aware UAV Tracking and Benchmarking**

SpoofTrackBench：用于欺骗感知无人机跟踪和基准的可解释人工智能 cs.CR

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22726v1) [paper-pdf](http://arxiv.org/pdf/2510.22726v1)

**Authors**: Van Le, Tan Le

**Abstract**: SpoofTrackBench is a reproducible, modular benchmark for evaluating adversarial robustness in real-time localization and tracking (RTLS) systems under radar spoofing. Leveraging the Hampton University Skyler Radar Sensor dataset, we simulate drift, ghost, and mirror-type spoofing attacks and evaluate tracker performance using both Joint Probabilistic Data Association (JPDA) and Global Nearest Neighbor (GNN) architectures. Our framework separates clean and spoofed detection streams, visualizes spoof-induced trajectory divergence, and quantifies assignment errors via direct drift-from-truth metrics. Clustering overlays, injection-aware timelines, and scenario-adaptive visualizations enable interpretability across spoof types and configurations. Evaluation figures and logs are auto-exported for reproducible comparison. SpoofTrackBench sets a new standard for open, ethical benchmarking of spoof-aware tracking pipelines, enabling rigorous cross-architecture analysis and community validation.

摘要: SpoofTrackBench是一个可重复的模块化基准，用于评估雷达欺骗下实时定位和跟踪（RTLS）系统的对抗鲁棒性。利用汉普顿大学斯凯勒雷达传感器数据集，我们模拟漂移、幽灵和镜子型欺骗攻击，并使用联合概率数据协会（JPDA）和全球最近邻（GNN）架构评估跟踪器的性能。我们的框架分离干净的检测流和欺骗的检测流，可视化欺骗引起的轨迹分歧，并通过直接的偏离真值指标量化分配误差。聚集叠加、注射感知时间线和场景自适应可视化实现了跨欺骗类型和配置的解释性。评估数字和日志会自动输出，以便进行可重复的比较。SpoofTrackBench为欺骗感知跟踪管道的开放、道德基准设定了新标准，实现了严格的跨架构分析和社区验证。



## **35. Measuring the (Un)Faithfulness of Concept-Based Explanations**

衡量基于概念的解释的（不）忠实性 cs.LG

Pre-print

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2504.10833v2) [paper-pdf](http://arxiv.org/pdf/2504.10833v2)

**Authors**: Shubham Kumar, Narendra Ahuja

**Abstract**: Post-hoc, unsupervised concept-based explanation methods (U-CBEMs) translate a vision model's internal reasoning into human-understandable concepts, leading to interpretable explanations. However, we find that many state-of-the-art (SOTA) U-CBEMs are not faithful: their concepts seem interpretable but fail to reproduce the model's predictions. We argue that this deficiency has gone unnoticed due to fragmented evaluation - each paper proposes its own faithfulness measure, with no measure-over-measure comparison or broad benchmarking. We close this gap by (i) organizing prior metrics in a unified framework, discussing their limitations, and identifying desiderata for a faithfulness measure; (ii) introducing the Surrogate Faithfulness (SURF) measure, which quantifies faithfulness via the predictive loss of a surrogate that maps explanations to the model's outputs; and (iii) delivering the first comprehensive U-CBEM faithfulness benchmark across diverse tasks and architectures. In a controlled setting, SURF outperforms prior faithfulness measures in measure-over-measure comparisons, and applying SURF to SOTA U-CBEMs reveals that many visually appealing U-CBEMs are surprisingly unfaithful. We demonstrate SURF applicability in two downstream settings - (i) faithfulness versus the number of concepts used in the explanation and (ii) U-CBEM robustness to adversarial attacks - underscoring SURF's value as a reliable faithfulness measure. Code to be released.

摘要: 事后、无监督的基于概念的解释方法（U-CBEM）将视觉模型的内部推理转化为人类可理解的概念，从而产生可解释的解释。然而，我们发现许多最先进的（SOTA）U-CBEM并不忠实：它们的概念似乎可以解释，但无法重现模型的预测。我们认为，由于评估分散，这一缺陷没有被注意到--每份论文都提出了自己的忠诚度衡量标准，没有衡量标准比较或广泛的基准。我们通过以下方式缩小这一差距：（i）在统一框架中组织先前的指标，讨论其局限性，并确定忠诚度衡量的愿望;（ii）引入代理忠诚度（SURF）衡量标准，该衡量标准通过预测性损失来量化忠诚度，该预测性损失将解释映射到模型的输出;（iii）跨不同任务和架构提供第一个全面的U-CBEM忠诚度基准。在受控环境中，SURF在测量与测量比较中优于之前的忠实度测量，将SURF应用于SOTA U-CBEM揭示了许多视觉上有吸引力的U-CBEM令人惊讶地不忠实。我们证明了SURF在两种下游环境中的适用性--（i）忠诚度与解释中使用的概念数量;（ii）U-CBEM对对抗性攻击的鲁棒性--强调了SURF作为可靠忠诚度指标的价值。待发布的代码。



## **36. If You Want to Be Robust, Be Wary of Initialization**

如果你想变得坚强，就要小心失败 cs.LG

Accepted at NeurIPS 2024

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22652v1) [paper-pdf](http://arxiv.org/pdf/2510.22652v1)

**Authors**: Sofiane Ennadir, Johannes F. Lutzeyer, Michalis Vazirgiannis, El Houcine Bergou

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable performance across a spectrum of graph-related tasks, however concerns persist regarding their vulnerability to adversarial perturbations. While prevailing defense strategies focus primarily on pre-processing techniques and adaptive message-passing schemes, this study delves into an under-explored dimension: the impact of weight initialization and associated hyper-parameters, such as training epochs, on a model's robustness. We introduce a theoretical framework bridging the connection between initialization strategies and a network's resilience to adversarial perturbations. Our analysis reveals a direct relationship between initial weights, number of training epochs and the model's vulnerability, offering new insights into adversarial robustness beyond conventional defense mechanisms. While our primary focus is on GNNs, we extend our theoretical framework, providing a general upper-bound applicable to Deep Neural Networks. Extensive experiments, spanning diverse models and real-world datasets subjected to various adversarial attacks, validate our findings. We illustrate that selecting appropriate initialization not only ensures performance on clean datasets but also enhances model robustness against adversarial perturbations, with observed gaps of up to 50\% compared to alternative initialization approaches.

摘要: 图形神经网络（GNN）在一系列图形相关任务中表现出了出色的性能，但人们仍然担心它们对对抗性扰动的脆弱性。虽然流行的防御策略主要关注预处理技术和自适应消息传递方案，但这项研究深入研究了一个未充分探索的维度：权重初始化和相关超参数（例如训练时期）对模型稳健性的影响。我们引入了一个理论框架，弥合初始化策略和网络对对抗性扰动的弹性之间的联系。我们的分析揭示了初始权重、训练时期数量和模型的脆弱性之间的直接关系，为超越传统防御机制的对抗鲁棒性提供了新的见解。虽然我们的主要重点是GNN，但我们扩展了我们的理论框架，提供了适用于深度神经网络的一般上限。跨越不同模型和遭受各种对抗攻击的现实世界数据集的广泛实验验证了我们的发现。我们说明，选择适当的初始化不仅可以确保干净数据集上的性能，而且还可以增强模型对对抗性扰动的鲁棒性，与替代初始化方法相比，观察到的差距高达50%。



## **37. Enhancing Graph Classification Robustness with Singular Pooling**

利用奇异池增强图分类的鲁棒性 cs.LG

Accepted at Neurips 2025

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22643v1) [paper-pdf](http://arxiv.org/pdf/2510.22643v1)

**Authors**: Sofiane Ennadir, Oleg Smirnov, Yassine Abbahaddou, Lele Cao, Johannes F. Lutzeyer

**Abstract**: Graph Neural Networks (GNNs) have achieved strong performance across a range of graph representation learning tasks, yet their adversarial robustness in graph classification remains underexplored compared to node classification. While most existing defenses focus on the message-passing component, this work investigates the overlooked role of pooling operations in shaping robustness. We present a theoretical analysis of standard flat pooling methods (sum, average and max), deriving upper bounds on their adversarial risk and identifying their vulnerabilities under different attack scenarios and graph structures. Motivated by these insights, we propose \textit{Robust Singular Pooling (RS-Pool)}, a novel pooling strategy that leverages the dominant singular vector of the node embedding matrix to construct a robust graph-level representation. We theoretically investigate the robustness of RS-Pool and interpret the resulting bound leading to improved understanding of our proposed pooling operator. While our analysis centers on Graph Convolutional Networks (GCNs), RS-Pool is model-agnostic and can be implemented efficiently via power iteration. Empirical results on real-world benchmarks show that RS-Pool provides better robustness than the considered pooling methods when subject to state-of-the-art adversarial attacks while maintaining competitive clean accuracy. Our code is publicly available at:\href{https://github.com/king/rs-pool}{https://github.com/king/rs-pool}.

摘要: 图神经网络（GNN）在一系列图表示学习任务中取得了很好的性能，但与节点分类相比，它们在图分类中的对抗鲁棒性仍然没有得到充分的研究。虽然大多数现有的防御措施都集中在消息传递组件上，但这项工作调查了池操作在塑造鲁棒性方面被忽视的作用。我们提出了一个理论分析标准的平池方法（总和，平均值和最大值），推导其对抗风险的上限，并确定其在不同的攻击场景和图形结构的漏洞。受这些见解的启发，我们提出了\textit{鲁棒奇异池（RS-Pool）}，一种新的池化策略，利用节点嵌入矩阵的主导奇异向量来构建鲁棒的图级表示。我们从理论上研究RS-Pool的稳健性并解释所得边界，从而提高对我们提出的池化操作符的理解。虽然我们的分析以图形卷积网络（GCN）为中心，但RS-Pool是模型不可知的，可以通过功率迭代有效实施。现实世界基准的经验结果表明，RS-Pool在受到最先进的对抗攻击时比考虑的池方法提供更好的鲁棒性，同时保持有竞争力的干净准确性。我们的代码可在以下网址公开获取：\href{https：//github.com/king/rs-pool}{https：//github.com/king/rs-pool}。



## **38. Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing**

Nes 2Net：基础模型驱动语音反欺骗的轻量级嵌套架构 eess.AS

Accepted to IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2504.05657v2) [paper-pdf](http://arxiv.org/pdf/2504.05657v2)

**Authors**: Tianchi Liu, Duc-Tuan Truong, Rohan Kumar Das, Kong Aik Lee, Haizhou Li

**Abstract**: Speech foundation models have significantly advanced various speech-related tasks by providing exceptional representation capabilities. However, their high-dimensional output features often create a mismatch with downstream task models, which typically require lower-dimensional inputs. A common solution is to apply a dimensionality reduction (DR) layer, but this approach increases parameter overhead, computational costs, and risks losing valuable information. To address these issues, we propose Nested Res2Net (Nes2Net), a lightweight back-end architecture designed to directly process high-dimensional features without DR layers. The nested structure enhances multi-scale feature extraction, improves feature interaction, and preserves high-dimensional information. We first validate Nes2Net on CtrSVDD, a singing voice deepfake detection dataset, and report a 22% performance improvement and an 87% back-end computational cost reduction over the state-of-the-art baseline. Additionally, extensive testing across four diverse datasets: ASVspoof 2021, ASVspoof 5, PartialSpoof, and In-the-Wild, covering fully spoofed speech, adversarial attacks, partial spoofing, and real-world scenarios, consistently highlights Nes2Net's superior robustness and generalization capabilities. The code package and pre-trained models are available at https://github.com/Liu-Tianchi/Nes2Net.

摘要: 语音基础模型通过提供出色的表示能力，显着推进了各种语音相关任务。然而，它们的多维输出特征通常会与下游任务模型产生不匹配，下游任务模型通常需要较低维度的输入。常见的解决方案是应用降维（DR）层，但这种方法增加了参数负担、计算成本，并存在丢失有价值信息的风险。为了解决这些问题，我们提出了Nested Res 2Net（Nes 2Net），这是一种轻量级的后台架构，旨在直接处理多维特征，而无需DR层。嵌套结构增强了多尺度特征提取，改善了特征交互，并保留了多维信息。我们首先在CtrSVD（歌唱声深度伪造检测数据集）上验证了Nes 2Net，并报告与最先进的基线相比，性能提高了22%，后台计算成本降低了87%。此外，对四个不同数据集进行了广泛的测试：ASVspoof 2021、ASVspoof 5、PartialSpoof和In-the-Wild，涵盖了完全欺骗的语音、对抗性攻击、部分欺骗和现实世界场景，一致强调了Nes 2Net卓越的鲁棒性和概括能力。代码包和预训练模型可在https://github.com/Liu-Tianchi/Nes2Net上获取。



## **39. Sentra-Guard: A Multilingual Human-AI Framework for Real-Time Defense Against Adversarial LLM Jailbreaks**

Sentra-Guard：用于实时防御对抗LLM越狱的多语言人工智能框架 cs.CR

11 pages, 5 figures. Preprint version under review in the area of  Artificial Intelligence (cs.AI)

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22628v1) [paper-pdf](http://arxiv.org/pdf/2510.22628v1)

**Authors**: Md. Mehedi Hasan, Ziaur Rahman, Rafid Mostafiz, Md. Abir Hossain

**Abstract**: This paper presents a real-time modular defense system named Sentra-Guard. The system detects and mitigates jailbreak and prompt injection attacks targeting large language models (LLMs). The framework uses a hybrid architecture with FAISS-indexed SBERT embedding representations that capture the semantic meaning of prompts, combined with fine-tuned transformer classifiers, which are machine learning models specialized for distinguishing between benign and adversarial language inputs. It identifies adversarial prompts in both direct and obfuscated attack vectors. A core innovation is the classifier-retriever fusion module, which dynamically computes context-aware risk scores that estimate how likely a prompt is to be adversarial based on its content and context. The framework ensures multilingual resilience with a language-agnostic preprocessing layer. This component automatically translates non-English prompts into English for semantic evaluation, enabling consistent detection across over 100 languages. The system includes a HITL feedback loop, where decisions made by the automated system are reviewed by human experts for continual learning and rapid adaptation under adversarial pressure. Sentra-Guard maintains an evolving dual-labeled knowledge base of benign and malicious prompts, enhancing detection reliability and reducing false positives. Evaluation results show a 99.96% detection rate (AUC = 1.00, F1 = 1.00) and an attack success rate (ASR) of only 0.004%. This outperforms leading baselines such as LlamaGuard-2 (1.3%) and OpenAI Moderation (3.7%). Unlike black-box approaches, Sentra-Guard is transparent, fine-tunable, and compatible with diverse LLM backends. Its modular design supports scalable deployment in both commercial and open-source environments. The system establishes a new state-of-the-art in adversarial LLM defense.

摘要: 本文提出了一种实时模块化防御系统Sentra-Guard。该系统检测并缓解针对大型语言模型（LLM）的越狱和提示注入攻击。该框架使用混合架构，该架构具有FAISS索引的SBERT嵌入表示，该表示捕获提示的语义含义，并结合了微调的Transformer分类器，后者是专门用于区分良性和对抗性语言输入的机器学习模型。它可以识别直接攻击向量和混淆攻击向量中的对抗性提示。一个核心创新是分类器-检索器融合模块，该模块动态计算上下文感知风险评分，根据其内容和上下文估计提示具有对抗性的可能性。该框架通过语言不可知的预处理层确保多语言弹性。该组件自动将非英语提示翻译成英语进行语义评估，从而实现对100多种语言的一致检测。该系统包括一个HITL反馈循环，其中自动化系统做出的决策由人类专家审查，以便在对抗压力下持续学习和快速适应。Sentra-Guard维护不断发展的良性和恶意提示的双标签知识库，增强检测可靠性并减少误报。评估结果显示，检测率为99.96%（AUR = 1.00，F1 = 1.00），攻击成功率（ASB）仅为0.004%。这优于LlamaGuard-2（1.3%）和OpenAI Moderation（3.7%）等领先基准。与黑匣子方法不同，Sentra-Guard是透明的、可微调的，并与各种LLM后台兼容。其模块化设计支持商业和开源环境中的可扩展部署。该系统在对抗性LLM辩护方面建立了新的最新水平。



## **40. Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents**

打破代理主干：评估人工智能代理中主干LLM的安全性 cs.CR

Julia Bazinska and Max Mathys contributed equally

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22620v1) [paper-pdf](http://arxiv.org/pdf/2510.22620v1)

**Authors**: Julia Bazinska, Max Mathys, Francesco Casucci, Mateo Rojas-Carulla, Xander Davies, Alexandra Souly, Niklas Pfister

**Abstract**: AI agents powered by large language models (LLMs) are being deployed at scale, yet we lack a systematic understanding of how the choice of backbone LLM affects agent security. The non-deterministic sequential nature of AI agents complicates security modeling, while the integration of traditional software with AI components entangles novel LLM vulnerabilities with conventional security risks. Existing frameworks only partially address these challenges as they either capture specific vulnerabilities only or require modeling of complete agents. To address these limitations, we introduce threat snapshots: a framework that isolates specific states in an agent's execution flow where LLM vulnerabilities manifest, enabling the systematic identification and categorization of security risks that propagate from the LLM to the agent level. We apply this framework to construct the $\operatorname{b}^3$ benchmark, a security benchmark based on 194331 unique crowdsourced adversarial attacks. We then evaluate 31 popular LLMs with it, revealing, among other insights, that enhanced reasoning capabilities improve security, while model size does not correlate with security. We release our benchmark, dataset, and evaluation code to facilitate widespread adoption by LLM providers and practitioners, offering guidance for agent developers and incentivizing model developers to prioritize backbone security improvements.

摘要: 由大型语言模型（LLM）支持的人工智能代理正在大规模部署，但我们缺乏对主干LLM的选择如何影响代理安全性的系统了解。人工智能代理的非确定性顺序性质使安全建模变得复杂，而传统软件与人工智能组件的集成则使新型LLM漏洞与传统安全风险纠缠在一起。现有的框架只能部分解决这些挑战，因为它们要么仅捕获特定的漏洞，要么需要对完整的代理进行建模。为了解决这些限制，我们引入了威胁快照：一个框架，可以隔离代理执行流程中LLM漏洞表现的特定状态，从而能够系统地识别和分类从LLM传播到代理级别的安全风险。我们应用此框架来构建$\operatorName{b}'#39; 3 $基准，这是一个基于194331个独特众包对抗攻击的安全基准。然后，我们用它评估了31种流行的LLM，揭示了增强的推理能力可以提高安全性，而模型大小与安全性无关。我们发布我们的基准、数据集和评估代码，以促进LLM提供商和从业者的广泛采用，为代理开发人员提供指导，并激励模型开发人员优先考虑主干安全改进。



## **41. Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks**

拉开帷幕：通过对比辅助网络的无监督对抗检测 cs.CV

Accepted for Oral Presentation at SafeMM-AI @ ICCV 2025 (Spotlight)

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2502.09110v3) [paper-pdf](http://arxiv.org/pdf/2502.09110v3)

**Authors**: Eylon Mizrahi, Raz Lapid, Moshe Sipper

**Abstract**: Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.

摘要: 深度学习模型广泛应用于安全关键应用中，但仍然容易受到对抗攻击--难以察觉的扰动，会显着降低模型性能。传统的防御机制主要专注于增强模型稳健性或独立检测对抗输入。在这项工作中，我们提出了一种通过对比辅助网络（U-CAN）的无监督对抗检测，以发现辅助特征表示中的对抗行为，而不需要对抗示例。U-CAN嵌入在目标模型的选定中间层中。这些辅助网络由投影层和基于ArcFace的线性层组成，可以细化特征表示，以更有效地区分良性输入和对抗输入。跨多个数据集（CIFAR-10、Mammals和ImageNet的一个子集）和架构（ResNet-50、VGG-16和ViT）的综合实验表明，我们的方法超越了现有的无监督对抗检测技术，在针对四种不同的攻击方法的情况下获得了优异的F1分数。提出的框架为增强深度学习系统的安全性和可靠性提供了可扩展且有效的解决方案。



## **42. FAARM: Firmware Attestation and Authentication Framework for Mali GPUs**

FAARM：马里图形处理器的硬件认证和认证框架 cs.CR

10 pages, 8 figures. Preprint version under review in the area of  Computer Security (cs.CR)

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22566v1) [paper-pdf](http://arxiv.org/pdf/2510.22566v1)

**Authors**: Md. Mehedi Hasan

**Abstract**: Recent work has revealed MOLE, the first practical attack to compromise GPU Trusted Execution Environments (TEEs), by injecting malicious firmware into the embedded Microcontroller Unit (MCU) of Arm Mali GPUs. By exploiting the absence of cryptographic verification during initialization, adversaries with kernel privileges can bypass memory protections, exfiltrate sensitive data at over 40 MB/s, and tamper with inference results, all with negligible runtime overhead. This attack surface affects commodity mobile SoCs and cloud accelerators, exposing a critical firmware-level trust gap in existing GPU TEE designs. To address this gap, this paper presents FAARM, a lightweight Firmware Attestation and Authentication framework that prevents MOLE-style firmware subversion. FAARM integrates digital signature verification at the EL3 secure monitor using vendor-signed firmware bundles and an on-device public key anchor. At boot, EL3 verifies firmware integrity and authenticity, enforces version checks, and locks the firmware region, eliminating both pre-verification and time-of-check-to-time-of-use (TOCTOU) attack vectors. We implement FAARM as a software-only prototype on a Mali GPU testbed, using a Google Colab-based emulation framework that models the firmware signing process, the EL1 to EL3 load path, and secure memory configuration. FAARM reliably detects and blocks malicious firmware injections, rejecting tampered images before use and denying overwrite attempts after attestation. Firmware verification incurs only 1.34 ms latency on average, demonstrating that strong security can be achieved with negligible overhead. FAARM thus closes a fundamental gap in shim-based GPU TEEs, providing a practical, deployable defense that raises the security baseline for both mobile and cloud GPU deployments.

摘要: 最近的工作揭示了MOLE，这是第一个危害图形处理器可信执行环境（TEE）的实际攻击，它通过将恶意硬件注入Arm Mali的嵌入式微控制器单元（MCU）中。通过利用初始化期间缺乏加密验证的机会，拥有内核特权的对手可以绕过内存保护，以超过40 MB/s的速度溢出敏感数据，并篡改推断结果，所有这些都只需忽略不计的运行时负载。这种攻击表面影响了商品移动SOC和云加速器，暴露了现有图形处理器EE设计中关键的公司级信任差距。为了解决这一差距，本文提出了FAARM，这是一个轻量级的硬件认证和认证框架，可以防止MOLE风格的硬件颠覆。FAARM使用供应商签名的固件包和设备上的公钥锚在EL 3安全监视器上集成了数字签名验证。在启动时，EL 3验证固件完整性和真实性，强制执行版本检查，并锁定固件区域，从而消除预验证和检查时间到使用时间（TOCTORM）攻击向量。我们在Mali GPU测试平台上将FAARM实现为纯软件原型，使用基于Google Colab的仿真框架，该框架对固件签名过程，EL 1到EL 3加载路径和安全内存配置进行建模。FAARM可靠地检测和阻止恶意固件注入，在使用前拒绝篡改图像，并在证明后拒绝覆盖尝试。硬件验证平均仅产生1.34 ms的延迟，这表明可以以微不足道的费用实现强大的安全性。因此，FAARM缩小了基于sham的图形处理器TEE的根本差距，提供了实用的、可部署的防御，提高了移动和云图形处理器部署的安全基线。



## **43. Cross-Paradigm Graph Backdoor Attacks with Promptable Subgraph Triggers**

具有可攻击子图触发器的跨范式图后门攻击 cs.CR

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22555v1) [paper-pdf](http://arxiv.org/pdf/2510.22555v1)

**Authors**: Dongyi Liu, Jiangtong Li, Dawei Cheng, Changjun Jiang

**Abstract**: Graph Neural Networks(GNNs) are vulnerable to backdoor attacks, where adversaries implant malicious triggers to manipulate model predictions.   Existing trigger generators are often simplistic in structure and overly reliant on specific features, confining them to a single graph learning paradigm, such as graph supervised learning, graph contrastive learning, or graph prompt learning.   This specialized design, which aligns the trigger with one learning objective, results in poor transferability when applied to other learning paradigms.   For instance, triggers generated for the graph supervised learning paradigm perform poorly when tested within graph contrastive learning or graph prompt learning environments.   Furthermore, these simple generators often fail to utilize complex structural information or node diversity within the graph data.   These constraints limit the attack success rates of such methods in general testing scenarios.   Therefore, to address these limitations, we propose Cross-Paradigm Graph Backdoor Attacks with Promptable Subgraph Triggers(CP-GBA), a new transferable graph backdoor attack that employs graph prompt learning(GPL) to train a set of universal subgraph triggers.   First, we distill a compact yet expressive trigger set from target graphs, which is structured as a queryable repository, by jointly enforcing class-awareness, feature richness, and structural fidelity.   Second, we conduct the first exploration of the theoretical transferability of GPL to train these triggers under prompt-based objectives, enabling effective generalization to diverse and unseen test-time paradigms.   Extensive experiments across multiple real-world datasets and defense scenarios show that CP-GBA achieves state-of-the-art attack success rates.

摘要: 图形神经网络（GNN）很容易受到后门攻击，对手会植入恶意触发器来操纵模型预测。   现有的触发生成器通常结构简单化，并且过度依赖特定特征，将它们限制在单个图学习范式中，例如图监督学习、图对比学习或图提示学习。   这种专门的设计将触发器与一个学习目标保持一致，但在应用于其他学习范式时会导致可移植性较差。   例如，当在图对比学习或图提示学习环境中进行测试时，为图监督学习范式生成的触发器表现不佳。   此外，这些简单的生成器通常无法利用图形数据中的复杂结构信息或节点多样性。   这些约束限制了此类方法在一般测试场景中的攻击成功率。   因此，为了解决这些限制，我们提出了具有可预测子图触发器的跨范式图后门攻击（CP-GBA），这是一种新的可转移图后门攻击，采用图提示学习（GPT）来训练一组通用子图触发器。   首先，我们从目标图中提取一个紧凑但富有表现力的触发集，通过联合实施类别意识、特征丰富性和结构保真度，该触发集被结构化为可查询的存储库。   其次，我们对GPT的理论可移植性进行了首次探索，以在基于预算的目标下训练这些触发器，从而能够有效地推广到多样化且不可见的测试时范式。   跨多个现实世界数据集和防御场景的广泛实验表明，CP-GBA实现了最先进的攻击成功率。



## **44. Security of Gradient Tracking Algorithms Against Malicious Agents**

针对恶意代理的梯度跟踪算法的安全性 eess.SY

under review

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2505.14473v2) [paper-pdf](http://arxiv.org/pdf/2505.14473v2)

**Authors**: Sribalaji C. Anand, Alexander J Gallo, Nicola Bastianello

**Abstract**: Consensus algorithms are fundamental to multi-agent distributed optimization, and their security under adversarial conditions is an active area of research. While prior works primarily establish conditions for successful global consensus under attack, little is known about system behavior when these conditions are violated. This paper addresses this gap by investigating the robustness of the Wang--Elia algorithm, which is a robust to noise version of gradient tracking algorithm, in the presence of malicious agents. We consider a network of agents collaboratively minimizing a global cost function, where a subset of agents may transmit faulty information to disrupt consensus. To quantify resilience, we formulate a security metric as an optimization problem, which is rooted in centralized attack detection literature. We provide a tractable reformulation of the optimization problem, and derive conditions under which the metric becomes unbounded, identifying undetectable attack signals that reveal inherent vulnerabilities. To facilitate design and analysis, we propose a well-posed variant of the metric and propose design methods to enhance network robustness against stealthy adversarial attacks. Numerical examples demonstrate the effectiveness of the proposed framework to enhance the resilience of multi-agent distributed optimization.

摘要: 共识算法是多代理分布式优化的基础，其在对抗条件下的安全性是一个活跃的研究领域。虽然先前的工作主要为在攻击下成功达成全球共识建立条件，但当这些条件被违反时，人们对系统行为知之甚少。本文通过研究Wang--Elia算法的鲁棒性来解决这一差距，该算法是在存在恶意代理的情况下对噪音鲁棒的梯度跟踪算法版本。我们考虑了一个代理网络，协作最小化全球成本函数，其中代理的子集可能会传输错误的信息以破坏共识。为了量化弹性，我们将安全指标制定为优化问题，该问题植根于集中式攻击检测文献。我们提供了优化问题的易于处理的重新表述，并推导出指标变得无界的条件，从而识别出揭示固有漏洞的无法检测到的攻击信号。为了促进设计和分析，我们提出了该指标的适定变体，并提出了设计方法来增强网络针对隐形对抗攻击的鲁棒性。数值例子证明了所提出的框架对于增强多代理分布式优化弹性的有效性。



## **45. Adapting Noise-Driven PUF and AI for Secure WBG ICS: A Proof-of-Concept Study**

调整噪音驱动的PFA和人工智能以实现安全WBG ICS：概念验证研究 cs.CR

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2510.22283v1) [paper-pdf](http://arxiv.org/pdf/2510.22283v1)

**Authors**: Devon A. Kelly, Christiana Chamon

**Abstract**: Wide-bandgap (WBG) technologies offer unprecedented improvements in power system efficiency, size, and performance, but also introduce unique sensor corruption and cybersecurity risks in industrial control systems (ICS), particularly due to high-frequency noise and sophisticated cyber-physical threats. This proof-of-concept (PoC) study demonstrates the adaptation of a noise-driven physically unclonable function (PUF) and machine learning (ML)-assisted anomaly detection framework to the demanding environment of WBG-based ICS sensor pathways. By extracting entropy from unavoidable WBG switching noise (up to 100 kHz) as a PUF source, and simultaneously using this noise as a real-time threat indicator, the proposed system unites hardware-level authentication and anomaly detection. Our approach integrates hybrid machine learning (ML) models with adaptive Bayesian filtering, providing robust and low-latency detection capabilities resilient to both natural electromagnetic interference (EMI) and active adversarial manipulation. Through detailed simulations of WBG modules under benign and attack scenarios--including EMI injection, signal tampering, and node impersonation--we achieve 95% detection accuracy and sub-millisecond processing latency. These results demonstrate the feasibility of physics-driven, dual-use noise exploitation as a scalable ICS defense primitive. Our findings lay the groundwork for next-generation security strategies that leverage inherent device characteristics, bridging hardware and artificial intelligence (AI) for enhanced protection of critical ICS infrastructure.

摘要: 宽带带隙（WBG）技术在电力系统效率、尺寸和性能方面提供了前所未有的改进，但也在工业控制系统（ICS）中引入了独特的传感器损坏和网络安全风险，特别是由于高频噪音和复杂的网络物理威胁。这项概念验证（RST）研究展示了噪音驱动的物理不可克隆功能（PFA）和机器学习（ML）辅助异常检测框架对基于WBG的ICS传感器路径的高要求环境的适应性。通过从不可避免的WBG开关噪音（高达100 GHz）中提取信息作为UF源，并同时使用该噪音作为实时威胁指示符，提出的系统将硬件级认证和异常检测结合起来。我们的方法将混合机器学习（ML）模型与自适应Bayesian过滤集成，提供稳健且低延迟的检测能力，能够抵御自然电磁干扰（EMI）和主动对抗操纵。通过在良性和攻击场景（包括EMI注入、信号篡改和节点模仿）下对WBG模块进行详细模拟，我们实现了95%的检测准确率和亚毫秒级的处理延迟。这些结果证明了物理驱动的两用噪音利用作为可扩展的ICS防御基元的可行性。我们的研究结果为下一代安全策略奠定了基础，这些策略利用固有的设备特征、连接硬件和人工智能（AI）来增强对关键ICS基础设施的保护。



## **46. SecureLearn -- An Attack-agnostic Defense for Multiclass Machine Learning Against Data Poisoning Attacks**

SecureLearn --针对数据中毒攻击的多类机器学习的攻击不可知防御 cs.CR

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2510.22274v1) [paper-pdf](http://arxiv.org/pdf/2510.22274v1)

**Authors**: Anum Paracha, Junaid Arshad, Mohamed Ben Farah, Khalid Ismail

**Abstract**: Data poisoning attacks are a potential threat to machine learning (ML) models, aiming to manipulate training datasets to disrupt their performance. Existing defenses are mostly designed to mitigate specific poisoning attacks or are aligned with particular ML algorithms. Furthermore, most defenses are developed to secure deep neural networks or binary classifiers. However, traditional multiclass classifiers need attention to be secure from data poisoning attacks, as these models are significant in developing multi-modal applications. Therefore, this paper proposes SecureLearn, a two-layer attack-agnostic defense to defend multiclass models from poisoning attacks. It comprises two components of data sanitization and a new feature-oriented adversarial training. To ascertain the effectiveness of SecureLearn, we proposed a 3D evaluation matrix with three orthogonal dimensions: data poisoning attack, data sanitization and adversarial training. Benchmarking SecureLearn in a 3D matrix, a detailed analysis is conducted at different poisoning levels (10%-20%), particularly analysing accuracy, recall, F1-score, detection and correction rates, and false discovery rate. The experimentation is conducted for four ML algorithms, namely Random Forest (RF), Decision Tree (DT), Gaussian Naive Bayes (GNB) and Multilayer Perceptron (MLP), trained with three public datasets, against three poisoning attacks and compared with two existing mitigations. Our results highlight that SecureLearn is effective against the provided attacks. SecureLearn has strengthened resilience and adversarial robustness of traditional multiclass models and neural networks, confirming its generalization beyond algorithm-specific defenses. It consistently maintained accuracy above 90%, recall and F1-score above 75%. For neural networks, SecureLearn achieved 97% recall and F1-score against all selected poisoning attacks.

摘要: 数据中毒攻击是对机器学习（ML）模型的潜在威胁，旨在操纵训练数据集以破坏其性能。现有的防御措施主要是为了减轻特定的中毒攻击或与特定的ML算法保持一致。此外，大多数防御措施都是为了保护深度神经网络或二进制分类器而开发的。然而，传统的多类分类器需要注意防止数据中毒攻击，因为这些模型在开发多模式应用程序方面非常重要。因此，本文提出SecureLearn，这是一种两层攻击不可知的防御，用于保护多类模型免受中毒攻击。它包括数据清理和新的面向特征的对抗训练两个部分。为了确定SecureLearn的有效性，我们提出了一个具有三个垂直维度的3D评估矩阵：数据中毒攻击、数据净化和对抗训练。在3D矩阵中对SecureLearn进行基准测试，在不同中毒水平（10%-20%）下进行详细分析，特别是分析准确性、召回率、F1评分、检测率和纠正率以及错误发现率。该实验是针对四种ML算法进行的，即随机森林（RF）、决策树（DT）、高斯朴素Bayes（GNB）和多层感知器（MLP），使用三个公共数据集进行训练，针对三种中毒攻击，并与两种现有的缓解措施进行了比较。我们的结果强调SecureLearn可以有效对抗所提供的攻击。SecureLearn增强了传统多类模型和神经网络的弹性和对抗鲁棒性，证实了其超越特定算法防御的普遍性。它始终保持在90%以上的准确率、召回率和F1评分超过75%。对于神经网络，SecureLearn针对所有选定的中毒攻击实现了97%的召回率和F1得分。



## **47. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

令人沮丧的简单但高效的攻击基线：针对GPT-4.5/4 o/o 1的强黑匣子模型的成功率超过90% cs.CV

NeurIPS 2025. Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2503.10635v2) [paper-pdf](http://arxiv.org/pdf/2503.10635v2)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against closed-source commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial black-box LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we propose to refine semantic clarity by encoding explicit semantic details within local regions, thus ensuring the capture of finer-grained features and inter-model transferability, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective baseline: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. While the naive source-target matching method has been utilized before in the literature, we are the first to provide a tight analysis, which establishes a close connection between perturbation optimization and semantics. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5/3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods with lower $\ell_1/\ell_2$ perturbations.

摘要: 尽管在开源大型视觉语言模型（LVLM）上表现出色，但基于传输的有针对性的攻击往往无法对封闭源商业LVLM进行攻击。分析失败的对抗性扰动表明，习得的扰动通常源于均匀分布，并且缺乏明确的语义细节，从而导致意外反应。语义信息的严重缺失导致商业黑匣子LVLM要么完全忽略扰动，要么误解其嵌入式语义，从而导致攻击失败。为了克服这些问题，我们建议通过在局部区域内编码显式的语义细节来完善语义清晰度，从而确保捕获更细粒度的特征和模型间的可移植性，并通过将修改集中在语义丰富的区域而不是统一应用它们。为了实现这一目标，我们提出了一个简单但高效的基线：在每个优化步骤中，通过受控的长宽比和比例随机裁剪对抗图像，调整大小，然后与嵌入空间中的目标图像对齐。虽然文献中以前曾使用过朴素的源目标匹配方法，但我们是第一个提供严密分析的人，该分析在扰动优化和语义之间建立了密切的联系。实验结果证实了我们的假设。我们用专注于关键区域的局部聚集扰动制作的对抗性示例表现出了令人惊讶的良好可移植性，包括GPT-4.5、GPT-4 o、Gemini-2.0-Flash、Claude-3.5/3.7-十四行诗，甚至还有像o 1、Claude-3.7-思考和Gemini-2.0-闪光思考这样的推理模型。我们的方法在GPT-4.5，4 o和o 1上实现了超过90%的成功率，显著优于所有现有的最先进的攻击方法，具有更低的$\ell_1/\ell_2$扰动。



## **48. Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization**

双流：通过野外级联流优化进行可转移的多目标、实例不可知攻击 cs.CV

Accepted at NeurIPS 2025

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2502.02096v3) [paper-pdf](http://arxiv.org/pdf/2502.02096v3)

**Authors**: Yixiao Chen, Shikun Sun, Jianshu Li, Ruoyu Li, Zhe Li, Junliang Xing

**Abstract**: Adversarial attacks are widely used to evaluate model robustness, and in black-box scenarios, the transferability of these attacks becomes crucial. Existing generator-based attacks have excellent generalization and transferability due to their instance-agnostic nature. However, when training generators for multi-target tasks, the success rate of transfer attacks is relatively low due to the limitations of the model's capacity. To address these challenges, we propose a novel Dual-Flow framework for multi-target instance-agnostic adversarial attacks, utilizing Cascading Distribution Shift Training to develop an adversarial velocity function. Extensive experiments demonstrate that Dual-Flow significantly improves transferability over previous multi-target generative attacks. For example, it increases the success rate from Inception-v3 to ResNet-152 by 34.58\%. Furthermore, our attack method shows substantially stronger robustness against defense mechanisms, such as adversarially trained models. The code of Dual-Flow is available at: $\href{https://github.com/Chyxx/Dual-Flow}{https://github.com/Chyxx/Dual-Flow}$.

摘要: 对抗性攻击被广泛用于评估模型稳健性，在黑匣子场景中，这些攻击的可转移性变得至关重要。现有的基于生成器的攻击由于其实例不可知的性质而具有出色的概括性和可移植性。然而，当训练多目标任务的生成器时，由于模型容量的限制，转移攻击的成功率相对较低。为了应对这些挑战，我们提出了一种新颖的双流框架，用于多目标实例不可知的对抗性攻击，利用级联分布转移训练来开发对抗性速度函数。大量实验表明，与之前的多目标生成式攻击相比，Dual-Flow显着提高了可移植性。例如，它将从Inception-v3到ResNet-152的成功率提高了34.58%。此外，我们的攻击方法对防御机制（例如对抗训练模型）表现出更强的鲁棒性。Dual-Flow的代码可在：$\href{https：//github.com/Chyxx/Dual-Flow}{https：//github.com/Chyxx/Dual-Flow}$。



## **49. Jailbreak Mimicry: Automated Discovery of Narrative-Based Jailbreaks for Large Language Models**

越狱模仿：大型语言模型基于叙事的越狱的自动发现 cs.CR

18 pages, 5 figures

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.22085v1) [paper-pdf](http://arxiv.org/pdf/2510.22085v1)

**Authors**: Pavlos Ntais

**Abstract**: Large language models (LLMs) remain vulnerable to sophisticated prompt engineering attacks that exploit contextual framing to bypass safety mechanisms, posing significant risks in cybersecurity applications. We introduce Jailbreak Mimicry, a systematic methodology for training compact attacker models to automatically generate narrative-based jailbreak prompts in a one-shot manner. Our approach transforms adversarial prompt discovery from manual craftsmanship into a reproducible scientific process, enabling proactive vulnerability assessment in AI-driven security systems. Developed for the OpenAI GPT-OSS-20B Red-Teaming Challenge, we use parameter-efficient fine-tuning (LoRA) on Mistral-7B with a curated dataset derived from AdvBench, achieving an 81.0% Attack Success Rate (ASR) against GPT-OSS-20B on a held-out test set of 200 items. Cross-model evaluation reveals significant variation in vulnerability patterns: our attacks achieve 66.5% ASR against GPT-4, 79.5% on Llama-3 and 33.0% against Gemini 2.5 Flash, demonstrating both broad applicability and model-specific defensive strengths in cybersecurity contexts. This represents a 54x improvement over direct prompting (1.5% ASR) and demonstrates systematic vulnerabilities in current safety alignment approaches. Our analysis reveals that technical domains (Cybersecurity: 93% ASR) and deception-based attacks (Fraud: 87.8% ASR) are particularly vulnerable, highlighting threats to AI-integrated threat detection, malware analysis, and secure systems, while physical harm categories show greater resistance (55.6% ASR). We employ automated harmfulness evaluation using Claude Sonnet 4, cross-validated with human expert assessment, ensuring reliable and scalable evaluation for cybersecurity red-teaming. Finally, we analyze failure mechanisms and discuss defensive strategies to mitigate these vulnerabilities in AI for cybersecurity.

摘要: 大型语言模型（LLM）仍然容易受到复杂的即时工程攻击，这些攻击利用上下文框架来绕过安全机制，从而给网络安全应用带来重大风险。我们引入了越狱模仿，这是一种系统性方法，用于训练紧凑的攻击者模型，以一次性方式自动生成基于叙述的越狱提示。我们的方法将对抗性即时发现从手工工艺转变为可重复的科学过程，从而在人工智能驱动的安全系统中实现主动的漏洞评估。我们为OpenAI GPT-OSS-20 B Red-Teaming Challenge而开发，在Mistral-7 B上使用参数高效微调（LoRA），采用源自AdvBench的精心策划数据集，在包含200个项目的测试集上实现了针对GPT-OSS-20 B的81.0%的攻击成功率（ASB）。跨模型评估揭示了漏洞模式的显著差异：我们的攻击对GPT-4实现了66.5%的ASR，对Llama-3实现了79.5%的ASR，对Gemini 2.5 Flash实现了33.0%的ASR，这表明了网络安全环境中的广泛适用性和特定于模型的防御优势。这比直接提示（1.5% ASR）提高了54倍，并表明当前安全调整方法存在系统性漏洞。我们的分析显示，技术领域（网络安全：93% ASR）和基于欺骗的攻击（欺诈：87.8% ASR）特别脆弱，突出了对AI集成威胁检测，恶意软件分析和安全系统的威胁，而物理伤害类别显示出更大的抵抗力（55.6% ASR）。我们使用Claude Sonnet 4进行自动危害性评估，并与人类专家评估进行交叉验证，确保对网络安全红队进行可靠和可扩展的评估。最后，我们分析了故障机制并讨论了缓解人工智能网络安全中这些漏洞的防御策略。



## **50. Toward Understanding the Transferability of Adversarial Suffixes in Large Language Models**

了解大型语言模型中对抗性后缀的可移植性 cs.CL

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.22014v1) [paper-pdf](http://arxiv.org/pdf/2510.22014v1)

**Authors**: Sarah Ball, Niki Hasrati, Alexander Robey, Avi Schwarzschild, Frauke Kreuter, Zico Kolter, Andrej Risteski

**Abstract**: Discrete optimization-based jailbreaking attacks on large language models aim to generate short, nonsensical suffixes that, when appended onto input prompts, elicit disallowed content. Notably, these suffixes are often transferable -- succeeding on prompts and models for which they were never optimized. And yet, despite the fact that transferability is surprising and empirically well-established, the field lacks a rigorous analysis of when and why transfer occurs. To fill this gap, we identify three statistical properties that strongly correlate with transfer success across numerous experimental settings: (1) how much a prompt without a suffix activates a model's internal refusal direction, (2) how strongly a suffix induces a push away from this direction, and (3) how large these shifts are in directions orthogonal to refusal. On the other hand, we find that prompt semantic similarity only weakly correlates with transfer success. These findings lead to a more fine-grained understanding of transferability, which we use in interventional experiments to showcase how our statistical analysis can translate into practical improvements in attack success.

摘要: 对大型语言模型的基于离散优化的越狱攻击旨在生成简短、无意义的后缀，当附加到输入提示时，会引发不允许的内容。值得注意的是，这些后缀通常是可移植的--在从未对其进行过优化的提示和模型上取得成功。然而，尽管可转移性令人惊讶并且在经验上得到了充分的证实，但该领域缺乏对何时和为何发生转移的严格分析。为了填补这一空白，我们确定了三个与众多实验环境中的转移成功密切相关的统计属性：（1）没有后缀的提示在多大程度上激活了模型的内部拒绝方向，（2）后缀引发了如何强烈的推动远离这个方向，以及（3）这些变化在与拒绝垂直的方向上有多大。另一方面，我们发现提示的语义相似性与转移成功仅弱相关。这些发现使我们对可移植性有了更细的理解，我们在干预实验中使用它来展示我们的统计分析如何转化为攻击成功的实际改进。




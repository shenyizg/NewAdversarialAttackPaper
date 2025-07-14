# Latest Adversarial Attack Papers
**update at 2025-07-14 09:58:56**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的弱到强越狱 cs.CL

ICML 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2401.17256v4) [paper-pdf](http://arxiv.org/pdf/2401.17256v4)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 大型语言模型（LLM）很容易受到越狱攻击，从而导致有害、不道德或有偏见的文本生成。然而，现有的越狱方法计算成本很高。本文中，我们提出了弱到强越狱攻击，这是一种针对对齐LLM的有效推理时间攻击，以产生有害文本。我们的关键直觉是基于这样的观察：越狱和对齐的模型仅在其初始解码分布上有所不同。从弱到强攻击的关键技术见解是使用两个较小的模型（一个安全的模型和一个不安全的模型）来对抗性地修改明显更大的安全模型的解码概率。我们评估了对来自3个组织的5个不同开源LLM的弱到强攻击。结果表明，我们的方法可以将两个数据集的未对准率提高到99%以上，每个示例只需向前传递一次。我们的研究揭示了在调整LLM时需要解决的紧迫安全问题。作为初步尝试，我们提出了一种防御策略来抵御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上获取



## **2. Entangled Threats: A Unified Kill Chain Model for Quantum Machine Learning Security**

纠缠威胁：量子机器学习安全的统一杀死链模型 quant-ph

Accepted for publication at IEEE International Conference on Quantum  Computing and Engineering (QCE) 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08623v1) [paper-pdf](http://arxiv.org/pdf/2507.08623v1)

**Authors**: Pascal Debus, Maximilian Wendlinger, Kilian Tscharke, Daniel Herr, Cedric Brügmann, Daniel Ohl de Mello, Juris Ulmanis, Alexander Erhard, Arthur Schmidt, Fabian Petsch

**Abstract**: Quantum Machine Learning (QML) systems inherit vulnerabilities from classical machine learning while introducing new attack surfaces rooted in the physical and algorithmic layers of quantum computing. Despite a growing body of research on individual attack vectors - ranging from adversarial poisoning and evasion to circuit-level backdoors, side-channel leakage, and model extraction - these threats are often analyzed in isolation, with unrealistic assumptions about attacker capabilities and system environments. This fragmentation hampers the development of effective, holistic defense strategies. In this work, we argue that QML security requires more structured modeling of the attack surface, capturing not only individual techniques but also their relationships, prerequisites, and potential impact across the QML pipeline. We propose adapting kill chain models, widely used in classical IT and cybersecurity, to the quantum machine learning context. Such models allow for structured reasoning about attacker objectives, capabilities, and possible multi-stage attack paths - spanning reconnaissance, initial access, manipulation, persistence, and exfiltration. Based on extensive literature analysis, we present a detailed taxonomy of QML attack vectors mapped to corresponding stages in a quantum-aware kill chain framework that is inspired by the MITRE ATLAS for classical machine learning. We highlight interdependencies between physical-level threats (like side-channel leakage and crosstalk faults), data and algorithm manipulation (such as poisoning or circuit backdoors), and privacy attacks (including model extraction and training data inference). This work provides a foundation for more realistic threat modeling and proactive security-in-depth design in the emerging field of quantum machine learning.

摘要: 量子机器学习（QML）系统继承了经典机器学习的漏洞，同时引入了植根于量子计算物理和算法层的新攻击表面。尽管对个体攻击载体的研究越来越多--从对抗性中毒和规避到电路级后门、侧通道泄漏和模型提取--但这些威胁通常是孤立地分析的，对攻击者的能力和系统环境做出了不切实际的假设。这种碎片化阻碍了有效、整体防御战略的制定。在这项工作中，我们认为QML安全需要对攻击表面进行更结构化的建模，不仅捕获单个技术，还捕获它们的关系、先决条件和整个QML管道的潜在影响。我们建议将广泛用于经典IT和网络安全的杀死链模型适应量子机器学习环境。此类模型允许对攻击者的目标、能力和可能的多阶段攻击路径进行结构化推理--跨越侦察、初始访问、操纵、持久性和溢出。基于广泛的文献分析，我们提出了映射到量子感知杀死链框架中相应阶段的QML攻击载体的详细分类，该框架的灵感来自经典机器学习的MITRE ATLAS。我们强调物理级别威胁（例如侧通道泄漏和串话故障）、数据和算法操纵（例如中毒或电路后门）以及隐私攻击（包括模型提取和训练数据推断）之间的相互依赖性。这项工作为量子机器学习新兴领域更现实的威胁建模和主动安全深度设计提供了基础。



## **3. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

LLM基于代理的完全计算机接管攻击的阴暗面 cs.CR

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.06850v3) [paper-pdf](http://arxiv.org/pdf/2507.06850v3)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.

摘要: 大型语言模型（LLM）代理和多代理系统的快速采用使自然语言处理和生成具有前所未有的能力。然而，这些系统引入了前所未有的安全漏洞，超出了传统的即时注入攻击的范围。本文首次对LLM代理进行了全面评估，作为攻击载体，这些攻击载体能够通过利用自主实体相互交互和影响的代理人工智能系统内的信任边界来实现完全的计算机接管。我们证明，对手可以利用三种不同的攻击表面--直接提示注入、RAG后门攻击和代理间信任利用--来强迫流行的LLM（包括GPT-4 o、Claude-4和Gemini-2.5）在受害者机器上自主安装和执行恶意软件。我们对17个最先进的LLM的评估揭示了一个令人震惊的漏洞层次结构：虽然41.2%的模型屈服于直接即时注入，但52.9%的模型容易受到RAG后门攻击，并且关键的82.4%可以通过代理间信任利用而受到损害。值得注意的是，我们发现成功抵抗直接恶意命令的LLM将在对等代理请求时执行相同的有效负载，这揭示了当前多代理安全模型中的一个根本缺陷。我们的研究结果表明，只有5.9%的测试模型（1/17）被证明能够抵抗所有攻击载体，其中大多数表现出依赖于上下文的安全行为，从而创建了可利用的盲点。我们的研究结果还强调了提高对LLM安全风险的认识和研究的必要性，这表明网络安全威胁的范式转变，人工智能工具本身成为复杂的攻击载体。



## **4. On the $(k,\ell)$-multiset anonymity measure for social graphs**

关于社交图的$（k，\ell）$-多集匿名性测量 math.CO

25 pages

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08433v1) [paper-pdf](http://arxiv.org/pdf/2507.08433v1)

**Authors**: Alejandro Estrada-Moreno, Elena Fernández, Dorota Kuziak, Manuel Muñoz-Márquez, Rolando Trujillo-Rasua, Ismael G. Yero

**Abstract**: The publication of social graphs must be preceded by a rigorous analysis of privacy threats against social graph users. When the threat comes from inside the social network itself, the threat is called an active attack, and the de-facto privacy measure used to quantify the resistance to such an attack is the $(k,\ell)$-anonymity. The original formulation of $(k,\ell)$-anonymity represents the adversary's knowledge as a vector of distances to the set of attacker nodes. In this article, we argue that such adversary is too strong when it comes to counteracting active attacks. We, instead, propose a new formulation where the adversary's knowledge is the multiset of distances to the set of attacker nodes. The goal of this article is to study the $(k,\ell)$-multiset anonymity from a graph theoretical point of view, while establishing its relationship to $(k,\ell)$-anonymity in one hand, and considering the $k$-multiset antiresolving sets as its theoretical frame, in a second one. That is, we prove properties of some graph families in relation to whether they contain a set of attacker nodes that breaks the $(k,\ell)$-multiset anonymity. From a practical point of view, we develop a linear programming formulation of the $k$-multiset antiresolving sets that allows us to calculate the resistance of social graphs against active attacks. This is useful for analysts who wish to know the level of privacy offered by a graph.

摘要: 发布社交图之前必须对社交图用户的隐私威胁进行严格分析。当威胁来自社交网络本身内部时，该威胁被称为主动攻击，用于量化此类攻击抵抗力的事实上的隐私指标是$（k，\ell）$-匿名性。$（k，\ell）$-匿名性的原始公式将对手的知识表示为与攻击者节点集的距离的载体。在本文中，我们认为这样的对手在对抗主动攻击方面过于强大。相反，我们提出了一种新的公式，其中对手的知识是到攻击者节点集的距离的多重集。本文的目标是从图形理论的角度研究$（k，\ell）$-多集匿名性，同时一方面建立它与$（k，\ell）$-匿名性的关系，并考虑$k$-多集反解析集作为其理论框架，在第二个框架中。也就是说，我们证明了一些图族的性质，该性质与它们是否包含一组破坏$（k，\ell）$-多集匿名性的攻击者节点有关。从实践的角度来看，我们开发了$k$-多集反解析集的线性规划公式，使我们能够计算社交图对主动攻击的抵抗力。这对于希望了解图表提供的隐私级别的分析师来说很有用。



## **5. Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving**

用于道路感知和物理可行自动驾驶的边界引导轨迹预测 cs.RO

Accepted in the 36th IEEE Intelligent Vehicles Symposium (IV 2025)

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2505.06740v2) [paper-pdf](http://arxiv.org/pdf/2505.06740v2)

**Authors**: Ahmed Abouelazm, Mianzhi Liu, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner

**Abstract**: Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66% to just 1%. These results highlight the effectiveness of our approach in generating feasible and robust predictions.

摘要: 准确预测周围道路使用者的轨迹对于安全高效的自动驾驶至关重要。虽然深度学习模型提高了性能，但在防止越野预测和确保运动学可行性方面仍然存在挑战。现有的方法包含道路感知模块并强制执行运动学约束，但缺乏合理性保证，并且经常在复杂性和灵活性方面引入权衡。本文提出了一种新颖的框架，将轨迹预测制定为由允许的驾驶方向及其边界引导的约束回归。使用代理的当前状态和高清地图，我们的方法定义有效边界，并通过训练网络学习左右边界多段线之间的叠加路径来确保道路预测。为了保证可行性，该模型预测加速度曲线，该曲线确定车辆沿着这些路径的行驶距离，同时遵守运动学约束。我们根据HTLR基线评估我们在Argoverse-2数据集上的方法。与HTLR相比，我们的方法显示基准指标略有下降，但显着改善了最终位移误差并消除了不可行的轨迹。此外，所提出的方法对不太常见的机动和不可见的非分布场景具有更好的通用性，将对抗性攻击下的越野率从66%降低到仅1%。这些结果凸显了我们的方法在生成可行且稳健的预测方面的有效性。



## **6. Minerva: A File-Based Ransomware Detector**

Minerva：基于文件的勒索软件检测器 cs.CR

Accepted for publication at The 20th ACM ASIA Conference on Computer  and Communications Security (ACM ASIACCS 2025), Meli\'a Hanoi

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2301.11050v4) [paper-pdf](http://arxiv.org/pdf/2301.11050v4)

**Authors**: Dorjan Hitaj, Giulio Pagnotta, Fabio De Gaspari, Lorenzo De Carli, Luigi V. Mancini

**Abstract**: Ransomware attacks have caused billions of dollars in damages in recent years, and are expected to cause billions more in the future. Consequently, significant effort has been devoted to ransomware detection and mitigation. Behavioral-based ransomware detection approaches have garnered considerable attention recently. These behavioral detectors typically rely on process-based behavioral profiles to identify malicious behaviors. However, with an increasing body of literature highlighting the vulnerability of such approaches to evasion attacks, a comprehensive solution to the ransomware problem remains elusive. This paper presents Minerva, a novel, robust approach to ransomware detection. Minerva is engineered to be robust by design against evasion attacks, with architectural and feature selection choices informed by their resilience to adversarial manipulation. We conduct a comprehensive analysis of Minerva across a diverse spectrum of ransomware types, encompassing unseen ransomware as well as variants designed specifically to evade Minerva. Our evaluation showcases the ability of Minerva to accurately identify ransomware, generalize to unseen threats, and withstand evasion attacks. Furthermore, over 99% of detected ransomware are identified within 0.52sec of activity, enabling the adoption of data loss prevention techniques with near-zero overhead.

摘要: 近年来，勒索软件攻击已造成数十亿美元的损失，预计未来还会造成数十亿美元的损失。因此，人们投入了大量精力来检测和缓解勒索软件。基于行为的勒索软件检测方法最近引起了相当大的关注。这些行为检测器通常依赖于基于流程的行为配置文件来识别恶意行为。然而，随着越来越多的文献强调这种方法对逃避攻击的脆弱性，勒索软件问题的全面解决方案仍然难以捉摸。本文介绍了Minerva，一种新颖的，强大的勒索软件检测方法。Minerva在设计上对规避攻击具有强大的鲁棒性，其架构和功能选择选择取决于其对对抗性操纵的弹性。我们对各种勒索软件类型的Minerva进行了全面分析，包括看不见的勒索软件以及专门为逃避Minerva而设计的变体。我们的评估展示了Minerva准确识别勒索软件、推广到不可见的威胁并抵御规避攻击的能力。此外，超过99%的检测到的勒索软件在活动后0.52秒内被识别出来，从而能够采用数据丢失预防技术，且费用接近零。



## **7. Towards Imperceptible JPEG Image Hiding: Multi-range Representations-driven Adversarial Stego Generation**

迈向不可感知的JPEG图像隐藏：多范围表示驱动的对抗性Stego生成 cs.CV

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08343v1) [paper-pdf](http://arxiv.org/pdf/2507.08343v1)

**Authors**: Junxue Yang, Xin Liao, Weixuan Tang, Jianhua Yang, Zheng Qin

**Abstract**: Deep hiding has been exploring the hiding capability of deep learning-based models, aiming to conceal image-level messages into cover images and reveal them from generated stego images. Existing schemes are easily detected by steganalyzers due to their large payloads and their limitation to feature extraction based solely on either pure convolution or pure transformer operators within a single range, as well as pixel-level loss constraints. To address the issue, in this paper, we introduce generation-based adversarial attacks into color JPEG image deep hiding and propose a multi-range representations-driven adversarial stego generation framework called MRAG from a steganalysis perspective. Specifically, we integrate the local-range neighbor reception characteristic of the convolution and the global-range dependency modeling of the transformer to construct MRAG. Meanwhile, we use the transformed images obtained through coarse-grained and fine-grained frequency decomposition as inputs, introducing multi-grained information. Furthermore, a features angle-norm disentanglement loss is designed to constrain the generated stegos closer to covers in the angle and norm space of the steganalyzer's classified features. Consequently, small yet effective adversarial perturbations can be injected into the process of generating stegos, ensuring that stegos maintain favorable secret restorability and imperceptibility. Extensive experiments demonstrate that MRAG can achieve state-of-the-art performance.

摘要: 深度隐藏一直在探索基于深度学习的模型的隐藏能力，旨在将图像级消息隐藏到封面图像中，并从生成的隐刻图像中揭示它们。现有的方案很容易被隐写分析器检测到，因为它们的有效负载大，而且它们对仅基于单一范围内的纯卷积或纯Transformer运算符的特征提取的限制，以及像素级损失约束。为了解决这个问题，在本文中，我们将基于生成的对抗性攻击引入到彩色JPEG图像深度隐藏中，并从隐写分析的角度提出了一个多范围表示驱动的对抗性隐写生成框架MRAG。具体来说，我们集成了卷积的局部范围邻居接收特性和Transformer的全球范围依赖性建模来构建MRAG。同时，我们使用粗粒度和细粒度频率分解获得的变换图像作为输入，引入多粒度信息。此外，设计了特征角度规范解纠缠损失，以将生成的隐写限制在隐写分析器分类特征的角度和规范空间中更接近覆盖。因此，可以将小而有效的对抗性扰动注入到生成隐果的过程中，确保隐果保持有利的秘密可感知性和不可感知性。大量实验表明MRAG可以实现最先进的性能。



## **8. Learning Robust Motion Skills via Critical Adversarial Attacks for Humanoid Robots**

通过关键对抗攻击学习仿人机器人稳健的运动技能 cs.RO

10 pages, 9 figures

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08303v1) [paper-pdf](http://arxiv.org/pdf/2507.08303v1)

**Authors**: Yang Zhang, Zhanxiang Cao, Buqing Nie, Haoyang Li, Yue Gao

**Abstract**: Humanoid robots show significant potential in daily tasks. However, reinforcement learning-based motion policies often suffer from robustness degradation due to the sim-to-real dynamics gap, thereby affecting the agility of real robots. In this work, we propose a novel robust adversarial training paradigm designed to enhance the robustness of humanoid motion policies in real worlds. The paradigm introduces a learnable adversarial attack network that precisely identifies vulnerabilities in motion policies and applies targeted perturbations, forcing the motion policy to enhance its robustness against perturbations through dynamic adversarial training. We conduct experiments on the Unitree G1 humanoid robot for both perceptive locomotion and whole-body control tasks. The results demonstrate that our proposed method significantly enhances the robot's motion robustness in real world environments, enabling successful traversal of challenging terrains and highly agile whole-body trajectory tracking.

摘要: 类人机器人在日常任务中显示出巨大的潜力。然而，基于强化学习的运动策略往往由于简单与真实的动态学差距而遭受鲁棒性下降，从而影响真实机器人的敏捷性。在这项工作中，我们提出了一种新颖的鲁棒对抗训练范式，旨在增强现实世界中人形运动策略的鲁棒性。该范式引入了一个可学习的对抗攻击网络，该网络精确识别运动策略中的漏洞并应用有针对性的扰动，迫使运动策略通过动态对抗训练增强其对扰动的鲁棒性。我们在Unitree G1人形机器人上进行了感知运动和全身控制任务的实验。结果表明，我们提出的方法显着增强了机器人在现实世界环境中的运动鲁棒性，能够成功穿越具有挑战性的地形和高度灵活的全身轨迹跟踪。



## **9. Lightweight Safety Guardrails via Synthetic Data and RL-guided Adversarial Training**

通过合成数据和RL引导的对抗训练的轻量级安全护栏 cs.LG

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08284v1) [paper-pdf](http://arxiv.org/pdf/2507.08284v1)

**Authors**: Aleksei Ilin, Gor Matevosyan, Xueying Ma, Vladimir Eremin, Suhaa Dada, Muqun Li, Riyaaz Shaik, Haluk Noyan Tokgozoglu

**Abstract**: We introduce a lightweight yet highly effective safety guardrail framework for language models, demonstrating that small-scale language models can achieve, and even surpass, the performance of larger counterparts in content moderation tasks. This is accomplished through high-fidelity synthetic data generation and adversarial training. The synthetic data generation process begins with human-curated seed data, which undergoes query augmentation and paraphrasing to create diverse and contextually rich examples. This augmented data is then subjected to multiple rounds of curation, ensuring high fidelity and relevance. Inspired by recent advances in the Generative Adversarial Network (GAN) architecture, our adversarial training employs reinforcement learning to guide a generator that produces challenging synthetic examples. These examples are used to fine-tune the safety classifier, enhancing its ability to detect and mitigate harmful content. Additionally, we incorporate strategies from recent research on efficient LLM training, leveraging the capabilities of smaller models to improve the performance of larger generative models. With iterative adversarial training and the generation of diverse, high-quality synthetic data, our framework enables small language models (SLMs) to serve as robust safety guardrails. This approach not only reduces computational overhead but also enhances resilience against adversarial attacks, offering a scalable and efficient solution for content moderation in AI systems.

摘要: 我们为语言模型引入了一个轻量级但高效的安全护栏框架，证明小规模语言模型可以在内容审核任务中实现甚至超越大型语言模型的性能。这是通过高保真合成数据生成和对抗训练来实现的。合成数据生成过程从人类精心策划的种子数据开始，该数据经过查询增强和解释，以创建多样化且上下文丰富的示例。然后，这些增强的数据经过多轮策展，确保高保真度和相关性。受生成对抗网络（GAN）架构最新进展的启发，我们的对抗训练采用强化学习来指导生成具有挑战性的合成示例的生成器。这些示例用于微调安全分类器，增强其检测和减轻有害内容的能力。此外，我们还结合了最近关于高效LLM培训的研究中的策略，利用较小模型的能力来提高较大生成模型的性能。通过迭代对抗训练和生成多样化、高质量的合成数据，我们的框架使小型语言模型（SLC）能够充当强大的安全护栏。这种方法不仅减少了计算负担，还增强了针对对抗攻击的弹性，为人工智能系统中的内容审核提供了可扩展且高效的解决方案。



## **10. Admissibility of Stein Shrinkage for Batch Normalization in the Presence of Adversarial Attacks**

对抗攻击下Stein收缩对批量归一化的可容许性 stat.ML

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08261v1) [paper-pdf](http://arxiv.org/pdf/2507.08261v1)

**Authors**: Sofia Ivolgina, P. Thomas Fletcher, Baba C. Vemuri

**Abstract**: Batch normalization (BN) is a ubiquitous operation in deep neural networks used primarily to achieve stability and regularization during network training. BN involves feature map centering and scaling using sample means and variances, respectively. Since these statistics are being estimated across the feature maps within a batch, this problem is ideally suited for the application of Stein's shrinkage estimation, which leads to a better, in the mean-squared-error sense, estimate of the mean and variance of the batch. In this paper, we prove that the Stein shrinkage estimator for the mean and variance dominates over the sample mean and variance estimators in the presence of adversarial attacks when modeling these attacks using sub-Gaussian distributions. This facilitates and justifies the application of Stein shrinkage to estimate the mean and variance parameters in BN and use it in image classification (segmentation) tasks with and without adversarial attacks. We present SOTA performance results using this Stein corrected batch norm in a standard ResNet architecture applied to the task of image classification using CIFAR-10 data, 3D CNN on PPMI (neuroimaging) data and image segmentation using HRNet on Cityscape data with and without adversarial attacks.

摘要: 批量正规化（BN）是深度神经网络中普遍存在的操作，主要用于在网络训练期间实现稳定性和正规化。BN涉及分别使用样本均值和方差对特征地图进行中心化和缩放。由于这些统计数据是在批次内的特征地图上估计的，因此这个问题非常适合应用斯坦的收缩估计，这可以在均方误差意义上对批次的均值和方差进行更好的估计。在本文中，我们证明，当使用亚高斯分布对这些攻击进行建模时，在存在对抗性攻击的情况下，均值和方差的Stein收缩估计器优于样本均值和方差估计器。这促进并证明了应用Stein收缩来估计BN中的均值和方差参数，并将其用于有和没有对抗攻击的图像分类（分割）任务。我们在标准ResNet架构中使用Stein纠正的批量规范来呈现SOTA性能结果，该架构应用于使用CIFAR-10数据的图像分类任务、PPMI（神经成像）数据上的3D CNN以及使用HRNet对Cityscape数据进行图像分割任务，有和没有对抗性攻击。



## **11. Pushing the Limits of Safety: A Technical Report on the ATLAS Challenge 2025**

突破安全极限：2025年ATLAS挑战赛技术报告 cs.CR

AdvML@CVPR Challenge Report

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2506.12430v2) [paper-pdf](http://arxiv.org/pdf/2506.12430v2)

**Authors**: Zonghao Ying, Siyang Wu, Run Hao, Peng Ying, Shixuan Sun, Pengyu Chen, Junze Chen, Hao Du, Kaiwen Shen, Shangkun Wu, Jiwei Wei, Shiyuan He, Yang Yang, Xiaohai Xu, Ke Ma, Qianqian Xu, Qingming Huang, Shi Lin, Xun Wang, Changting Lin, Meng Han, Yilei Jiang, Siqi Lai, Yaozhi Zheng, Yifei Song, Xiangyu Yue, Zonglei Jing, Tianyuan Zhang, Zhilei Zhu, Aishan Liu, Jiakai Wang, Siyuan Liang, Xianglong Kong, Hainan Li, Junjie Mu, Haotong Qin, Yue Yu, Lei Chen, Felix Juefei-Xu, Qing Guo, Xinyun Chen, Yew Soon Ong, Xianglong Liu, Dawn Song, Alan Yuille, Philip Torr, Dacheng Tao

**Abstract**: Multimodal Large Language Models (MLLMs) have enabled transformative advancements across diverse applications but remain susceptible to safety threats, especially jailbreak attacks that induce harmful outputs. To systematically evaluate and improve their safety, we organized the Adversarial Testing & Large-model Alignment Safety Grand Challenge (ATLAS) 2025}. This technical report presents findings from the competition, which involved 86 teams testing MLLM vulnerabilities via adversarial image-text attacks in two phases: white-box and black-box evaluations. The competition results highlight ongoing challenges in securing MLLMs and provide valuable guidance for developing stronger defense mechanisms. The challenge establishes new benchmarks for MLLM safety evaluation and lays groundwork for advancing safer multimodal AI systems. The code and data for this challenge are openly available at https://github.com/NY1024/ATLAS_Challenge_2025.

摘要: 多模式大型语言模型（MLLM）在不同的应用程序中实现了变革性的进步，但仍然容易受到安全威胁，尤其是引发有害输出的越狱攻击。为了系统地评估和提高其安全性，我们组织了对抗性测试和大模型对齐安全大挑战赛（ATLAS）2025。本技术报告介绍了比赛的结果，其中86个团队通过对抗性图像文本攻击分两个阶段测试MLLM漏洞：白盒和黑匣子评估。竞赛结果凸显了确保MLLM方面持续存在的挑战，并为开发更强大的防御机制提供了宝贵的指导。该挑战为MLLM安全评估建立了新的基准，并为推进更安全的多模式人工智能系统奠定了基础。此挑战的代码和数据可在https://github.com/NY1024/ATLAS_Challenge_2025上公开获取。



## **12. A Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking**

动态Stackelberg游戏框架，用于针对LLM越狱的大型人工智能防御 cs.AI

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08207v1) [paper-pdf](http://arxiv.org/pdf/2507.08207v1)

**Authors**: Zhengye Han, Quanyan Zhu

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, the challenge of jailbreaking, where adversaries manipulate the models to bypass safety mechanisms, has become a significant concern. This paper presents a dynamic Stackelberg game framework to model the interactions between attackers and defenders in the context of LLM jailbreaking. The framework treats the prompt-response dynamics as a sequential extensive-form game, where the defender, as the leader, commits to a strategy while anticipating the attacker's optimal responses. We propose a novel agentic AI solution, the "Purple Agent," which integrates adversarial exploration and defensive strategies using Rapidly-exploring Random Trees (RRT). The Purple Agent actively simulates potential attack trajectories and intervenes proactively to prevent harmful outputs. This approach offers a principled method for analyzing adversarial dynamics and provides a foundation for mitigating the risk of jailbreaking.

摘要: 随着大型语言模型（LLM）越来越多地部署在关键应用程序中，越狱的挑战（对手操纵模型以绕过安全机制）已成为一个重大问题。本文提出了一个动态Stackelberg博弈框架，来建模LLM越狱背景下攻击者和防御者之间的互动。该框架将预算-响应动态视为一个顺序扩展形式的游戏，其中防御者作为领导者，承诺采取策略，同时预测攻击者的最佳响应。我们提出了一种新型的代理人工智能解决方案，即“紫色代理”，它使用快速探索随机树（RTI）集成了对抗性探索和防御策略。Purple Agent主动模拟潜在的攻击轨迹，并主动干预以防止有害输出。这种方法提供了一种分析对抗动态的原则性方法，并为减轻越狱风险提供了基础。



## **13. Beyond the Worst Case: Extending Differential Privacy Guarantees to Realistic Adversaries**

超越最坏情况：将差异隐私保证扩展到现实对手 cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08158v1) [paper-pdf](http://arxiv.org/pdf/2507.08158v1)

**Authors**: Marika Swanberg, Meenatchi Sundaram Muthu Selva Annamalai, Jamie Hayes, Borja Balle, Adam Smith

**Abstract**: Differential Privacy (DP) is a family of definitions that bound the worst-case privacy leakage of a mechanism. One important feature of the worst-case DP guarantee is it naturally implies protections against adversaries with less prior information, more sophisticated attack goals, and complex measures of a successful attack. However, the analytical tradeoffs between the adversarial model and the privacy protections conferred by DP are not well understood thus far. To that end, this work sheds light on what the worst-case guarantee of DP implies about the success of attackers that are more representative of real-world privacy risks.   In this paper, we present a single flexible framework that generalizes and extends the patchwork of bounds on DP mechanisms found in prior work. Our framework allows us to compute high-probability guarantees for DP mechanisms on a large family of natural attack settings that previous bounds do not capture. One class of such settings is the approximate reconstruction of multiple individuals' data, such as inferring nearly entire columns of a tabular data set from noisy marginals and extracting sensitive information from DP-trained language models.   We conduct two empirical case studies to illustrate the versatility of our bounds and compare them to the success of state-of-the-art attacks. Specifically, we study attacks that extract non-uniform PII from a DP-trained language model, as well as multi-column reconstruction attacks where the adversary has access to some columns in the clear and attempts to reconstruct the remaining columns for each person's record. We find that the absolute privacy risk of attacking non-uniform data is highly dependent on the adversary's prior probability of success. Our high probability bounds give us a nuanced understanding of the privacy leakage of DP mechanisms in a variety of previously understudied attack settings.

摘要: 差异隐私（DP）是一系列定义，限制了机制的最坏情况隐私泄露。最坏情况DP保证的一个重要特征是，它自然意味着针对先验信息较少、攻击目标更复杂且成功攻击措施复杂的对手提供保护。然而，迄今为止，对抗模型和DP赋予的隐私保护之间的分析权衡还没有得到很好的理解。为此，这项工作揭示了DP的最坏情况保证对更能代表现实世界隐私风险的攻击者的成功意味着什么。   在本文中，我们提出了一个灵活的框架，该框架概括和扩展了先前工作中发现的DP机制边界的拼凑。我们的框架允许我们在以前的界限无法捕捉的一大系列自然攻击设置上计算DP机制的高概率保证。一类此类设置是多个人数据的大致重建，例如从有噪的边缘推断表格数据集的几乎整个列，并从DP训练的语言模型中提取敏感信息。   我们进行了两个实证案例研究，以说明我们边界的多功能性，并将它们与最先进的攻击的成功进行比较。具体来说，我们研究了从DP训练的语言模型中提取非均匀PRI的攻击，以及多列重建攻击，其中对手可以以明文方式访问某些列并试图为每个人的记录重建剩余列。我们发现，攻击非均匀数据的绝对隐私风险高度取决于对手的先验成功概率。我们的高概率界限让我们对各种以前未充分研究的攻击环境中DP机制的隐私泄露有了细致入微的了解。



## **14. Hedge Funds on a Swamp: Analyzing Patterns, Vulnerabilities, and Defense Measures in Blockchain Bridges [Experiment, Analysis & Benchmark]**

沼泽上的对冲基金：分析区块链桥梁中的模式、漏洞和防御措施[实验、分析和基准] cs.ET

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.06156v2) [paper-pdf](http://arxiv.org/pdf/2507.06156v2)

**Authors**: Poupak Azad, Jiahua Xu, Yebo Feng, Preston Strowbridge, Cuneyt Akcora

**Abstract**: Blockchain bridges have become essential infrastructure for enabling interoperability across different blockchain networks, with more than $24B monthly bridge transaction volume. However, their growing adoption has been accompanied by a disproportionate rise in security breaches, making them the single largest source of financial loss in Web3. For cross-chain ecosystems to be robust and sustainable, it is essential to understand and address these vulnerabilities. In this study, we present a comprehensive systematization of blockchain bridge design and security. We define three bridge security priors, formalize the architectural structure of 13 prominent bridges, and identify 23 attack vectors grounded in real-world blockchain exploits. Using this foundation, we evaluate 43 representative attack scenarios and introduce a layered threat model that captures security failures across source chain, off-chain, and destination chain components.   Our analysis at the static code and transaction network levels reveals recurring design flaws, particularly in access control, validator trust assumptions, and verification logic, and identifies key patterns in adversarial behavior based on transaction-level traces. To support future development, we propose a decision framework for bridge architecture design, along with defense mechanisms such as layered validation and circuit breakers. This work provides a data-driven foundation for evaluating bridge security and lays the groundwork for standardizing resilient cross-chain infrastructure.

摘要: 区块链桥梁已成为实现不同区块链网络互操作性的重要基础设施，每月桥梁交易量超过240亿美元。然而，随着它们的日益普及，安全漏洞也不成比例地增加，使它们成为Web 3中最大的财务损失来源。为了实现跨链生态系统的稳健和可持续发展，了解和解决这些脆弱性至关重要。在这项研究中，我们对区块链桥梁设计和安全进行了全面的系统化。我们定义了三个桥梁安全先验，正式确定了13个突出桥梁的架构结构，并确定了23个基于现实世界区块链漏洞的攻击向量。在此基础上，我们评估了43种有代表性的攻击场景，并引入了一个分层的威胁模型，该模型可以捕获源链、链下和目标链组件的安全故障。   我们在静态代码和交易网络层面的分析揭示了反复出现的设计缺陷，特别是在访问控制、验证者信任假设和验证逻辑方面，并根据交易级跟踪识别了对抗行为的关键模式。为了支持未来的发展，我们提出了一个决策框架的桥梁架构设计，以及防御机制，如分层验证和断路器。这项工作为评估桥梁安全性提供了数据驱动的基础，并为标准化弹性跨链基础设施奠定了基础。



## **15. KeyDroid: A Large-Scale Analysis of Secure Key Storage in Android Apps**

KeyDroid：Android应用程序中安全密钥存储的大规模分析 cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07927v1) [paper-pdf](http://arxiv.org/pdf/2507.07927v1)

**Authors**: Jenny Blessing, Ross J. Anderson, Alastair R. Beresford

**Abstract**: Most contemporary mobile devices offer hardware-backed storage for cryptographic keys, user data, and other sensitive credentials. Such hardware protects credentials from extraction by an adversary who has compromised the main operating system, such as a malicious third-party app. Since 2011, Android app developers can access trusted hardware via the Android Keystore API. In this work, we conduct the first comprehensive survey of hardware-backed key storage in Android devices. We analyze 490 119 Android apps, collecting data on how trusted hardware is used by app developers (if used at all) and cross-referencing our findings with sensitive user data collected by each app, as self-reported by developers via the Play Store's data safety labels.   We find that despite industry-wide initiatives to encourage adoption, 56.3% of apps self-reporting as processing sensitive user data do not use Android's trusted hardware capabilities at all, while just 5.03% of apps collecting some form of sensitive data use the strongest form of trusted hardware, a secure element distinct from the main processor. To better understand the potential downsides of using secure hardware, we conduct the first empirical analysis of trusted hardware performance in mobile devices, measuring the runtime of common cryptographic operations across both software- and hardware-backed keystores. We find that while hardware-backed key storage using a coprocessor is viable for most common cryptographic operations, secure elements capable of preventing more advanced attacks make performance infeasible for symmetric encryption with non-negligible payloads and any kind of asymmetric encryption.

摘要: 大多数当代移动设备都为密钥、用户数据和其他敏感凭证提供硬件支持的存储。此类硬件可以保护凭据免受危害主操作系统的对手（例如恶意第三方应用程序）提取。自2011年以来，Android应用程序开发人员可以通过Android Keystore API访问受信任的硬件。在这项工作中，我们对Android设备中硬件支持的密钥存储进行了首次全面调查。我们分析了490 119个Android应用程序，收集有关应用程序开发人员如何使用可信硬件（如果有的话）的数据，并将我们的调查结果与每个应用程序收集的敏感用户数据进行交叉引用，这些数据由开发人员通过Play Store的数据安全标签自我报告。   我们发现，尽管行业范围内采取了鼓励采用的举措，但56.3%自我报告处理敏感用户数据的应用程序根本不使用Android的受信任硬件功能，而收集某种形式敏感数据的应用程序中，只有5.03%使用最强形式的受信任硬件，这是一种与主处理器不同的安全元素。为了更好地了解使用安全硬件的潜在缺点，我们对移动设备中的可信硬件性能进行了首次实证分析，测量了软件和硬件支持的密钥库中常见加密操作的运行时间。我们发现，虽然使用协处理器的硬件支持密钥存储对于大多数常见的加密操作来说是可行的，但能够防止更高级攻击的安全元素使得具有不可忽视的有效负载的对称加密和任何类型的非对称加密的性能不可行。



## **16. Bayes-Nash Generative Privacy Against Membership Inference Attacks**

针对会员推断攻击的Bayes-Nash生成隐私 cs.CR

arXiv admin note: substantial text overlap with arXiv:2406.01811

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2410.07414v5) [paper-pdf](http://arxiv.org/pdf/2410.07414v5)

**Authors**: Tao Zhang, Rajagopal Venkatesaramani, Rajat K. De, Bradley A. Malin, Yevgeniy Vorobeychik

**Abstract**: Membership inference attacks (MIAs) pose significant privacy risks by determining whether individual data is in a dataset. While differential privacy (DP) mitigates these risks, it has limitations including limited resolution in expressing privacy-utility tradeoffs and intractable sensitivity calculations for tight guarantees. We propose a game-theoretic framework modeling privacy protection as a Bayesian game between defender and attacker, where privacy loss corresponds to the attacker's membership inference ability. To address strategic complexity, we represent the defender's mixed strategy as a neural network generator mapping private datasets to public representations (e.g., noisy statistics) and the attacker's strategy as a discriminator making membership claims. This \textit{general-sum Generative Adversarial Network} trains iteratively through alternating updates, yielding \textit{Bayes-Nash Generative Privacy (BNGP)} strategies. BNGP avoids worst-case privacy proofs such as sensitivity calculations, supports correlated mechanism compositions, handles heterogeneous attacker preferences. Empirical studies on sensitive dataset summary statistics show our approach significantly outperforms state-of-the-art methods by generating stronger attacks and achieving better privacy-utility tradeoffs.

摘要: 成员资格推断攻击（MIA）通过确定单个数据是否位于数据集中而构成重大隐私风险。虽然差异隐私（DP）可以减轻这些风险，但它也有局限性，包括表达隐私与公用事业权衡的分辨率有限以及严格保证的棘手敏感性计算。我们提出了一个博弈论框架，将隐私保护建模为防御者和攻击者之间的Bayesian博弈，其中隐私损失对应于攻击者的成员资格推断能力。为了解决战略复杂性，我们将防御者的混合策略表示为神经网络生成器，将私人数据集映射到公共表示（例如，有噪音的统计数据）以及攻击者的策略作为一个制造会员资格的声明。此\texttit {general-sum Generative Adversarial Network}通过交替更新迭代训练，产生\texttit {Bayes-Nash Generative Privacy（BNGP）}策略。BCGP避免了最坏情况的隐私证明，例如敏感度计算，支持相关机制组合，处理异类攻击者偏好。对敏感数据集摘要统计数据的实证研究表明，我们的方法通过产生更强的攻击并实现更好的隐私与公用事业权衡而显着优于最先进的方法。



## **17. Identifying the Smallest Adversarial Load Perturbations that Render DC-OPF Infeasible**

识别导致DC-OPF不可行的最小对抗负载扰动 eess.SY

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07850v1) [paper-pdf](http://arxiv.org/pdf/2507.07850v1)

**Authors**: Samuel Chevalier, William A. Wheeler

**Abstract**: What is the globally smallest load perturbation that renders DC-OPF infeasible? Reliably identifying such "adversarial attack" perturbations has useful applications in a variety of emerging grid-related contexts, including machine learning performance verification, cybersecurity, and operational robustness of power systems dominated by stochastic renewable energy resources. In this paper, we formulate the inherently nonconvex adversarial attack problem by applying a parameterized version of Farkas' lemma to a perturbed set of DC-OPF equations. Since the resulting formulation is very hard to globally optimize, we also propose a parameterized generation control policy which, when applied to the primal DC-OPF problem, provides solvability guarantees. Together, these nonconvex problems provide guaranteed upper and lower bounds on adversarial attack size; by combining them into a single optimization problem, we can efficiently "squeeze" these bounds towards a common global solution. We apply these methods on a range of small- to medium-sized test cases from PGLib, benchmarking our results against the best adversarial attack lower bounds provided by Gurobi 12.0's spatial Branch and Bound solver.

摘要: 导致DC-OPF不可行的全球最小负载扰动是多少？可靠地识别这种“对抗攻击”扰动在各种新兴的电网相关环境中具有有用的应用，包括机器学习性能验证、网络安全和由随机可再生能源主导的电力系统的运营稳健性。在本文中，我们通过将Farkas引理的参数化版本应用于一组受干扰的DC-OPF方程，来阐述固有非凸对抗攻击问题。由于所得公式很难全局优化，我们还提出了一种参数化发电控制策略，当应用于原始DC-OPF问题时，该策略提供了可解性保证。这些非凸问题共同提供了对抗性攻击规模的有保证的上下限;通过将它们组合到单个优化问题中，我们可以有效地“挤压”这些界限以获得共同的全局解决方案。我们将这些方法应用于PGLib的一系列中小规模测试用例，并根据Guesthouse 12.0的空间Branch and Bound求解器提供的最佳对抗攻击下限对我们的结果进行基准测试。



## **18. "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models**

“我很坏”：在音频语言模型中解释秘密、普遍和稳健的音频越狱 cs.LG

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2502.00718v2) [paper-pdf](http://arxiv.org/pdf/2502.00718v2)

**Authors**: Isha Gupta, David Khachaturov, Robert Mullins

**Abstract**: The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.

摘要: 多模式大型语言模型的兴起引入了创新的人机交互范式，但也给机器学习安全带来了重大挑战。由于口语交流的直观性，音频语言模型（ILM）尤其重要，但人们对其失败模式知之甚少。本文探讨了针对ILM的音频越狱，重点关注它们绕过对齐机制的能力。我们构建了跨越提示、任务甚至基本音频样本的对抗性扰动，展示了音频模式中的第一次普遍越狱，并表明这些在模拟的现实世界条件下仍然有效。除了证明攻击可行性之外，我们还分析了ILM如何解释这些音频对抗示例，并揭示它们来编码难以察觉的第一人称有毒语音-这表明用于引发有毒输出的最有效的干扰专门嵌入了音频信号中的语言特征。这些结果对于理解多模式模型中不同模式之间的相互作用具有重要意义，并为增强对抗性音频攻击的防御提供了可行的见解。



## **19. SCOOTER: A Human Evaluation Framework for Unrestricted Adversarial Examples**

SCOTER：无限制对抗性例子的人类评估框架 cs.CV

42 pages, 16 figures, 11 tables, Under Review, Code:  https://github.com/DrenFazlija/Scooter, Data:  https://doi.org/10.5281/zenodo.15771501

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07776v1) [paper-pdf](http://arxiv.org/pdf/2507.07776v1)

**Authors**: Dren Fazlija, Monty-Maximilian Zühlke, Johanna Schrader, Arkadij Orlov, Clara Stein, Iyiola E. Olatunji, Daniel Kudenko

**Abstract**: Unrestricted adversarial attacks aim to fool computer vision models without being constrained by $\ell_p$-norm bounds to remain imperceptible to humans, for example, by changing an object's color. This allows attackers to circumvent traditional, norm-bounded defense strategies such as adversarial training or certified defense strategies. However, due to their unrestricted nature, there are also no guarantees of norm-based imperceptibility, necessitating human evaluations to verify just how authentic these adversarial examples look. While some related work assesses this vital quality of adversarial attacks, none provide statistically significant insights. This issue necessitates a unified framework that supports and streamlines such an assessment for evaluating and comparing unrestricted attacks. To close this gap, we introduce SCOOTER - an open-source, statistically powered framework for evaluating unrestricted adversarial examples. Our contributions are: $(i)$ best-practice guidelines for crowd-study power, compensation, and Likert equivalence bounds to measure imperceptibility; $(ii)$ the first large-scale human vs. model comparison across 346 human participants showing that three color-space attacks and three diffusion-based attacks fail to produce imperceptible images. Furthermore, we found that GPT-4o can serve as a preliminary test for imperceptibility, but it only consistently detects adversarial examples for four out of six tested attacks; $(iii)$ open-source software tools, including a browser-based task template to collect annotations and analysis scripts in Python and R; $(iv)$ an ImageNet-derived benchmark dataset containing 3K real images, 7K adversarial examples, and over 34K human ratings. Our findings demonstrate that automated vision systems do not align with human perception, reinforcing the need for a ground-truth SCOOTER benchmark.

摘要: 无限制的对抗攻击旨在愚弄计算机视觉模型，而不受$\ell_p$-norm边界的约束，以保持人类不可感知，例如，通过改变对象的颜色。这使得攻击者能够规避传统的、规范有限的防御策略，例如对抗性训练或认证的防御策略。然而，由于其不受限制的性质，也无法保证基于规范的不可感知性，因此需要进行人类评估来验证这些对抗性例子看起来有多真实。虽然一些相关工作评估了对抗性攻击的这一重要性质，但没有一项工作提供统计上显着的见解。这个问题需要一个统一的框架来支持和简化这种评估，以评估和比较无限制的攻击。为了缩小这一差距，我们引入了SCOOTER -一个开源的，统计动力的框架，用于评估不受限制的对抗性示例。我们的贡献是：$（i）$用于测量不可感知性的群体研究功率、补偿和李克特等效界限的最佳实践指南; $（ii）$在346名人类参与者中进行的第一次大规模人类与模型比较，表明三种颜色空间攻击和三种基于扩散的攻击无法产生不可感知的图像。此外，我们发现GPT-4 o可以作为不可感知性的初步测试，但它只能持续检测六种测试攻击中的四种攻击的对抗性示例; $（iii）$开源软件工具，包括基于浏览器的任务模板，用于收集Python和R中的注释和分析脚本;$（iv）$ImageNet衍生的基准数据集，包含3 K真实图像，7 K对抗性示例和超过34 K的人类评分。我们的研究结果表明，自动视觉系统不符合人类的感知，加强了对地面实况滑板车基准的需求。



## **20. Rainbow Artifacts from Electromagnetic Signal Injection Attacks on Image Sensors**

图像传感器电磁信号注入攻击产生的彩虹伪影 cs.CR

5 pages, 4 figures

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07773v1) [paper-pdf](http://arxiv.org/pdf/2507.07773v1)

**Authors**: Youqian Zhang, Xinyu Ji, Zhihao Wang, Qinhong Jiang

**Abstract**: Image sensors are integral to a wide range of safety- and security-critical systems, including surveillance infrastructure, autonomous vehicles, and industrial automation. These systems rely on the integrity of visual data to make decisions. In this work, we investigate a novel class of electromagnetic signal injection attacks that target the analog domain of image sensors, allowing adversaries to manipulate raw visual inputs without triggering conventional digital integrity checks. We uncover a previously undocumented attack phenomenon on CMOS image sensors: rainbow-like color artifacts induced in images captured by image sensors through carefully tuned electromagnetic interference. We further evaluate the impact of these attacks on state-of-the-art object detection models, showing that the injected artifacts propagate through the image signal processing pipeline and lead to significant mispredictions. Our findings highlight a critical and underexplored vulnerability in the visual perception stack, highlighting the need for more robust defenses against physical-layer attacks in such systems.

摘要: 图像传感器是各种安全和安全关键系统的组成部分，包括监控基础设施、自动驾驶汽车和工业自动化。这些系统依赖视觉数据的完整性来做出决策。在这项工作中，我们研究了一类新型的电磁信号注入攻击，这些攻击针对图像传感器的模拟域，允许对手在不触发传统数字完整性检查的情况下操纵原始视觉输入。我们发现了一种以前未记录的针对互补性图像传感器的攻击现象：通过精心调整的电磁干扰，图像传感器捕获的图像中引发彩虹般的色彩伪影。我们进一步评估了这些攻击对最先进对象检测模型的影响，表明注入的伪影通过图像信号处理管道传播并导致严重的误预测。我们的研究结果凸显了视觉感知堆栈中一个关键且未充分探索的漏洞，凸显了对此类系统中物理层攻击的更强大防御的必要性。



## **21. TRIX- Trading Adversarial Fairness via Mixed Adversarial Training**

TRIX-通过混合对抗培训实现对抗公平交易 cs.LG

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07768v1) [paper-pdf](http://arxiv.org/pdf/2507.07768v1)

**Authors**: Tejaswini Medi, Steffen Jung, Margret Keuper

**Abstract**: Adversarial Training (AT) is a widely adopted defense against adversarial examples. However, existing approaches typically apply a uniform training objective across all classes, overlooking disparities in class-wise vulnerability. This results in adversarial unfairness: classes with well distinguishable features (strong classes) tend to become more robust, while classes with overlapping or shared features(weak classes) remain disproportionately susceptible to adversarial attacks. We observe that strong classes do not require strong adversaries during training, as their non-robust features are quickly suppressed. In contrast, weak classes benefit from stronger adversaries to effectively reduce their vulnerabilities. Motivated by this, we introduce TRIX, a feature-aware adversarial training framework that adaptively assigns weaker targeted adversaries to strong classes, promoting feature diversity via uniformly sampled targets, and stronger untargeted adversaries to weak classes, enhancing their focused robustness. TRIX further incorporates per-class loss weighting and perturbation strength adjustments, building on prior work, to emphasize weak classes during the optimization. Comprehensive experiments on standard image classification benchmarks, including evaluations under strong attacks such as PGD and AutoAttack, demonstrate that TRIX significantly improves worst-case class accuracy on both clean and adversarial data, reducing inter-class robustness disparities, and preserves overall accuracy. Our results highlight TRIX as a practical step toward fair and effective adversarial defense.

摘要: 对抗性训练（AT）是一种广泛采用的针对对抗性例子的防御方法。然而，现有的方法通常在所有班级中应用统一的培训目标，忽略了班级脆弱性的差异。这会导致对抗性不公平：具有良好可区分特征的类（强类）往往会变得更加稳健，而具有重叠或共享特征的类（弱类）仍然不成比例地容易受到对抗性攻击。我们观察到，强类在训练期间不需要强对手，因为它们的非稳健特征很快就会被抑制。相比之下，弱势阶层从更强大的对手中受益，从而有效减少他们的脆弱性。受此启发，我们引入了TRIX，这是一个特征感知的对抗训练框架，它自适应地将较弱的目标对手分配给强类，通过均匀采样的目标促进特征多样性，并将较强的非目标对手分配给弱类，增强其集中的鲁棒性。TRIX在先前工作的基础上进一步结合了按类别的损失加权和扰动强度调整，以在优化期间强调弱类别。对标准图像分类基准的全面实验（包括PVD和AutoAttack等强攻击下的评估）表明，TRIX显着提高了干净和对抗数据的最坏情况类别准确性，减少了类别间稳健性差异，并保持了整体准确性。我们的结果强调TRIX是实现公平有效对抗性防御的实际步骤。



## **22. One Object, Multiple Lies: A Benchmark for Cross-task Adversarial Attack on Unified Vision-Language Models**

一个对象，多个谎言：统一视觉语言模型跨任务对抗攻击的基准 cs.CV

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07709v1) [paper-pdf](http://arxiv.org/pdf/2507.07709v1)

**Authors**: Jiale Zhao, Xinyang Jiang, Junyao Gao, Yuhao Xue, Cairong Zhao

**Abstract**: Unified vision-language models(VLMs) have recently shown remarkable progress, enabling a single model to flexibly address diverse tasks through different instructions within a shared computational architecture. This instruction-based control mechanism creates unique security challenges, as adversarial inputs must remain effective across multiple task instructions that may be unpredictably applied to process the same malicious content. In this paper, we introduce CrossVLAD, a new benchmark dataset carefully curated from MSCOCO with GPT-4-assisted annotations for systematically evaluating cross-task adversarial attacks on unified VLMs. CrossVLAD centers on the object-change objective-consistently manipulating a target object's classification across four downstream tasks-and proposes a novel success rate metric that measures simultaneous misclassification across all tasks, providing a rigorous evaluation of adversarial transferability. To tackle this challenge, we present CRAFT (Cross-task Region-based Attack Framework with Token-alignment), an efficient region-centric attack method. Extensive experiments on Florence-2 and other popular unified VLMs demonstrate that our method outperforms existing approaches in both overall cross-task attack performance and targeted object-change success rates, highlighting its effectiveness in adversarially influencing unified VLMs across diverse tasks.

摘要: 统一视觉语言模型（VLM）最近取得了显着的进展，使单个模型能够通过共享计算架构内的不同指令灵活地处理不同的任务。这种基于描述的控制机制带来了独特的安全挑战，因为对抗性输入必须在多个任务指令中保持有效，这些指令可能不可预测地应用于处理相同的恶意内容。本文中，我们介绍了CrossVLDA，这是一个由MSCOCO精心策划的新基准数据集，具有GPT-4辅助注释，用于系统性评估对统一VLM的跨任务对抗攻击。CrossVRAD以对象更改目标为中心--在四个下游任务中一致地操纵目标对象的分类--并提出了一种新颖的成功率指标，该指标可以衡量所有任务中的同时错误分类，从而对对抗性可移植性进行严格评估。为了应对这一挑战，我们提出了CRAFT（具有令牌对齐的跨任务区域攻击框架），这是一种高效的以区域为中心的攻击方法。对Florence-2和其他流行的统一VLM的广泛实验表明，我们的方法在总体跨任务攻击性能和有针对性的对象更改成功率方面优于现有方法，凸显了其在对不同任务中的统一VLM进行不利影响方面的有效性。



## **23. Impact Assessment of Cyberattacks in Inverter-Based Microgrids**

基于逆变器的微电网网络攻击的影响评估 eess.SY

IEEE Workshop on the Electronic Grid (eGrid 2025)

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2504.05592v2) [paper-pdf](http://arxiv.org/pdf/2504.05592v2)

**Authors**: Kerd Topallaj, Colin McKerrell, Suraj Ramanathan, Ioannis Zografopoulos

**Abstract**: In recent years, the evolution of modern power grids has been driven by the growing integration of remotely controlled grid assets. Although Distributed Energy Resources (DERs) and Inverter-Based Resources (IBRs) enhance operational efficiency, they also introduce cybersecurity risks. The remote accessibility of such critical grid components creates entry points for attacks that adversaries could exploit, posing threats to the stability of the system. To evaluate the resilience of energy systems under such threats, this study employs real-time simulation and a modified version of the IEEE 39-bus system that incorporates a Microgrid (MG) with solar-based IBR. The study assesses the impact of remote attacks impacting the MG stability under different levels of IBR penetration through hardware-in-the-loop (HIL) simulations. Namely, we analyze voltage, current, and frequency profiles before, during, and after cyberattack-induced disruptions. The results demonstrate that real-time HIL testing is a practical approach to uncover potential risks and develop robust mitigation strategies for resilient MG operations.

摘要: 近年来，远程控制电网资产的日益整合推动了现代电网的发展。尽管分布式能源资源（BER）和基于逆变器的资源（IBR）提高了运营效率，但它们也带来了网络安全风险。此类关键网格组件的远程访问性为对手可能利用的攻击创建了切入点，从而对系统的稳定性构成威胁。为了评估能源系统在此类威胁下的弹性，本研究采用了实时模拟和IEEE 39节点系统的修改版本，该系统结合了微电网（MG）和基于太阳能的IBR。该研究通过硬件在环（HIL）模拟评估了远程攻击在不同IBR渗透水平下对MG稳定性的影响。也就是说，我们分析网络攻击引发的中断之前、期间和之后的电压、电流和频率分布。结果表明，实时HIL测试是发现潜在风险并为有弹性的MG运营制定稳健的缓解策略的实用方法。



## **24. ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense**

ARBoids：采用Boids模型的自适应剩余强化学习用于协作多USV目标防御 cs.LG

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2502.18549v2) [paper-pdf](http://arxiv.org/pdf/2502.18549v2)

**Authors**: Jiyue Tao, Tongsheng Shen, Dexin Zhao, Feitian Zhang

**Abstract**: The target defense problem (TDP) for unmanned surface vehicles (USVs) concerns intercepting an adversarial USV before it breaches a designated target region, using one or more defending USVs. A particularly challenging scenario arises when the attacker exhibits superior maneuverability compared to the defenders, significantly complicating effective interception. To tackle this challenge, this letter introduces ARBoids, a novel adaptive residual reinforcement learning framework that integrates deep reinforcement learning (DRL) with the biologically inspired, force-based Boids model. Within this framework, the Boids model serves as a computationally efficient baseline policy for multi-agent coordination, while DRL learns a residual policy to adaptively refine and optimize the defenders' actions. The proposed approach is validated in a high-fidelity Gazebo simulation environment, demonstrating superior performance over traditional interception strategies, including pure force-based approaches and vanilla DRL policies. Furthermore, the learned policy exhibits strong adaptability to attackers with diverse maneuverability profiles, highlighting its robustness and generalization capability. The code of ARBoids will be released upon acceptance of this letter.

摘要: 无人水面航行器（USV）的目标防御问题（SDP）涉及在敌方USV突破指定目标区域之前使用一辆或多辆防御USV拦截其。当攻击者与防御者相比表现出更好的机动性时，就会出现一种特别具有挑战性的情况，从而使有效拦截变得非常复杂。为了应对这一挑战，这封信介绍了ARBoids，这是一种新型的自适应剩余强化学习框架，它将深度强化学习（DRL）与生物启发的、基于力的Boids模型集成在一起。在此框架中，Boids模型充当多智能体协调的计算高效基线策略，而DRL则学习剩余策略以自适应地细化和优化防御者的动作。所提出的方法在高保真Gazebo模拟环境中得到了验证，证明了优于传统拦截策略（包括纯粹基于力量的方法和普通DRL策略）的性能。此外，学习到的策略对具有不同机动性特征的攻击者表现出很强的适应性，凸显了其鲁棒性和概括能力。ARBoids的代码将在接受本信函后发布。



## **25. Autonomous AI-based Cybersecurity Framework for Critical Infrastructure: Real-Time Threat Mitigation**

关键基础设施的基于人工智能的自主网络安全框架：实时威胁缓解 cs.CR

7 pages, IEEE conference

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07416v1) [paper-pdf](http://arxiv.org/pdf/2507.07416v1)

**Authors**: Jenifer Paulraj, Brindha Raghuraman, Nagarani Gopalakrishnan, Yazan Otoum

**Abstract**: Critical infrastructure systems, including energy grids, healthcare facilities, transportation networks, and water distribution systems, are pivotal to societal stability and economic resilience. However, the increasing interconnectivity of these systems exposes them to various cyber threats, including ransomware, Denial-of-Service (DoS) attacks, and Advanced Persistent Threats (APTs). This paper examines cybersecurity vulnerabilities in critical infrastructure, highlighting the threat landscape, attack vectors, and the role of Artificial Intelligence (AI) in mitigating these risks. We propose a hybrid AI-driven cybersecurity framework to enhance real-time vulnerability detection, threat modelling, and automated remediation. This study also addresses the complexities of adversarial AI, regulatory compliance, and integration. Our findings provide actionable insights to strengthen the security and resilience of critical infrastructure systems against emerging cyber threats.

摘要: 关键基础设施系统，包括电网、医疗保健设施、交通网络和供水系统，对于社会稳定和经济弹性至关重要。然而，这些系统日益增强的互连性使它们面临各种网络威胁，包括勒索软件、拒绝服务（DPS）攻击和高级持续性威胁（APT）。本文研究了关键基础设施中的网络安全漏洞，重点介绍了威胁格局、攻击载体以及人工智能（AI）在减轻这些风险方面的作用。我们提出了一个混合人工智能驱动的网络安全框架，以增强实时漏洞检测、威胁建模和自动化修复。这项研究还解决了对抗性人工智能、监管合规性和整合的复杂性。我们的研究结果提供了可操作的见解，以加强关键基础设施系统针对新兴网络威胁的安全性和弹性。



## **26. Phishing Detection in the Gen-AI Era: Quantized LLMs vs Classical Models**

Gen-AI时代的网络钓鱼检测：量化LLM与经典模型 cs.CR

8 Pages, IEEE Conference

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07406v1) [paper-pdf](http://arxiv.org/pdf/2507.07406v1)

**Authors**: Jikesh Thapa, Gurrehmat Chahal, Serban Voinea Gabreanu, Yazan Otoum

**Abstract**: Phishing attacks are becoming increasingly sophisticated, underscoring the need for detection systems that strike a balance between high accuracy and computational efficiency. This paper presents a comparative evaluation of traditional Machine Learning (ML), Deep Learning (DL), and quantized small-parameter Large Language Models (LLMs) for phishing detection. Through experiments on a curated dataset, we show that while LLMs currently underperform compared to ML and DL methods in terms of raw accuracy, they exhibit strong potential for identifying subtle, context-based phishing cues. We also investigate the impact of zero-shot and few-shot prompting strategies, revealing that LLM-rephrased emails can significantly degrade the performance of both ML and LLM-based detectors. Our benchmarking highlights that models like DeepSeek R1 Distill Qwen 14B (Q8_0) achieve competitive accuracy, above 80%, using only 17GB of VRAM, supporting their viability for cost-efficient deployment. We further assess the models' adversarial robustness and cost-performance tradeoffs, and demonstrate how lightweight LLMs can provide concise, interpretable explanations to support real-time decision-making. These findings position optimized LLMs as promising components in phishing defence systems and offer a path forward for integrating explainable, efficient AI into modern cybersecurity frameworks.

摘要: 网络钓鱼攻击变得越来越复杂，这凸显了对在高准确性和计算效率之间取得平衡的检测系统的需求。本文对传统机器学习（ML）、深度学习（DL）和用于网络钓鱼检测的量化小参数大型语言模型（LLM）进行了比较评估。通过对精心策划的数据集的实验，我们表明，虽然LLM目前在原始准确性方面表现不佳ML和DL方法，但它们在识别微妙的、基于上下文的网络钓鱼线索方面表现出强大的潜力。我们还研究了零激发和少激发策略的影响，揭示了LLM重新措辞的电子邮件会显着降低ML和基于LLM的检测器的性能。我们的基准测试强调，DeepSeek R1 Distill Qwen 14 B（Q8_0）等型号仅使用17 GB VRAM即可实现80%以上的竞争准确性，支持其具有成本效益的部署可行性。我们进一步评估了模型的对抗稳健性和成本-性能权衡，并展示了轻量级LLM如何提供简洁、可解释的解释以支持实时决策。这些发现将优化的LLM定位为网络钓鱼防御系统中有前途的组件，并为将可解释、高效的人工智能集成到现代网络安全框架中提供了前进的道路。



## **27. A Cryptographic Perspective on Mitigation vs. Detection in Machine Learning**

从密码学角度看机器学习中的缓解与检测 cs.LG

28 pages

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2504.20310v2) [paper-pdf](http://arxiv.org/pdf/2504.20310v2)

**Authors**: Greg Gluch, Shafi Goldwasser

**Abstract**: In this paper, we initiate a cryptographically inspired theoretical study of detection versus mitigation of adversarial inputs produced by attackers on Machine Learning algorithms during inference time.   We formally define defense by detection (DbD) and defense by mitigation (DbM). Our definitions come in the form of a 3-round protocol between two resource-bounded parties: a trainer/defender and an attacker. The attacker aims to produce inference-time inputs that fool the training algorithm. We define correctness, completeness, and soundness properties to capture successful defense at inference time while not degrading (too much) the performance of the algorithm on inputs from the training distribution.   We first show that achieving DbD and achieving DbM are equivalent for ML classification tasks. Surprisingly, this is not the case for ML generative learning tasks, where there are many possible correct outputs for each input. We show a separation between DbD and DbM by exhibiting two generative learning tasks for which it is possible to defend by mitigation but it is provably impossible to defend by detection. The mitigation phase uses significantly less computational resources than the initial training algorithm. In the first learning task we consider sample complexity as the resource and in the second the time complexity. The first result holds under the assumption that the Identity-Based Fully Homomorphic Encryption (IB-FHE), publicly-verifiable zero-knowledge Succinct Non-Interactive Arguments of Knowledge (zk-SNARK), and Strongly Unforgeable Signatures exist. The second result assumes the existence of Non-Parallelizing Languages with Average-Case Hardness (NPL) and Incrementally-Verifiable Computation (IVC) and IB-FHE.

摘要: 在本文中，我们启动了一项受密码启发的理论研究，研究攻击者在推理时间内对机器学习算法产生的对抗输入的检测与缓解。   我们正式定义了检测防御（GbD）和缓解防御（GbM）。我们的定义以两个资源有限方之间的三轮协议的形式出现：训练者/防御者和攻击者。攻击者的目标是产生欺骗训练算法的推断时输入。我们定义了正确性、完整性和稳健性属性，以在推理时捕获成功的防御，同时不会降低（太多）算法在训练分布输入上的性能。   我们首先表明，实现GbD和实现GbM对于ML分类任务来说是等效的。令人惊讶的是，ML生成式学习任务的情况并非如此，其中每个输入都有许多可能的正确输出。我们通过展示两个生成性学习任务来展示GbD和GbM之间的分离，对于这些任务，可以通过缓解来防御，但事实证明不可能通过检测来防御。缓解阶段使用的计算资源比初始训练算法少得多。在第一个学习任务中，我们将样本复杂性视为资源，在第二个学习任务中，我们将样本复杂性视为资源。第一个结果成立的假设是基于身份的完全同质加密（IB-FHE）、可公开验证的零知识简洁非交互式知识参数（zk-SNARK）和强不可伪造签名。第二个结果假设存在具有平均情况硬度（NPL）、增量可验证计算（IVC）和IB-FHE的非并行化语言。



## **28. Adversarial Defenses via Vector Quantization**

通过载体量化的对抗防御 cs.LG

This is the author-accepted version of our paper published in  Neurocomputing. The final published version is available at:  https://doi.org/10.1016/j.neucom.2025.130703

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2305.13651v2) [paper-pdf](http://arxiv.org/pdf/2305.13651v2)

**Authors**: Zhiyi Dong, Yongyi Mao

**Abstract**: Adversarial attacks pose significant challenges to the robustness of modern deep neural networks in computer vision, and defending these networks against adversarial attacks has attracted intense research efforts. Among various defense strategies, preprocessing-based defenses are practically appealing since there is no need to train the network under protection. However, such approaches typically do not achieve comparable robustness as other methods such as adversarial training. In this paper, we propose a novel framework for preprocessing-based defenses, where a vector quantizer is used as a preprocessor. This framework, inspired by and extended from Randomized Discretization (RandDisc), is theoretically principled by rate-distortion theory: indeed, RandDisc may be viewed as a scalar quantizer, and rate-distortion theory suggests that such quantization schemes are inferior to vector quantization. In our framework, the preprocessing vector quantizer treats the input image as a collection of patches and finds a set of representative patches based on the patch distributions; each original patch is then modified according to the representative patches close to it. We present two lightweight defenses in this framework, referred to as patched RandDisc (pRD) and sliding-window RandDisc (swRD), where the patches are disjoint in the former and overlapping in the latter. We show that vector-quantization-based defenses have certifiable robust accuracy and that pRD and swRD demonstrate state-of-the-art performances, surpassing RandDisc by a large margin. Notably, the proposed defenses possess the obfuscated gradients property. Our experiments however show that pRD and swRD remain effective under the STE and EOT attacks, which are designed specifically for defenses with gradient obfuscation. ...

摘要: 对抗性攻击对计算机视觉中现代深度神经网络的鲁棒性构成了重大挑战，保护这些网络免受对抗性攻击吸引了大量的研究工作。在各种防御策略中，基于预处理的防御实际上很有吸引力，因为不需要训练受保护的网络。然而，此类方法通常无法实现与对抗训练等其他方法相当的鲁棒性。在本文中，我们提出了一种基于预处理的防御的新颖框架，其中使用一个载体量化器作为预处理器。这个框架受到随机离散化（RandDisc）的启发和扩展，在理论上以率失真理论为原则：事实上，RandDisc可以被视为一个纯量量化器，而率失真理论表明这种量化方案不如向量量化。在我们的框架中，预处理载体量化器将输入图像视为补丁的集合，并根据补丁分布找到一组代表性补丁;然后根据接近它的代表性补丁来修改每个原始补丁。我们在这个框架中提出了两种轻量级防御，称为补丁RandDisc（pRD）和滑动窗口RandDisc（swRD），其中斑块在前者中是不相交的，而在后者中是重叠的。我们表明，基于量化的防御具有可认证的稳健准确性，并且pRD和swRD表现出最先进的性能，大幅超过RandDisc。值得注意的是，拟议的防御具有模糊梯度属性。然而，我们的实验表明，pRD和swRD在STE和OT攻击下仍然有效，STE和OT攻击是专门为具有梯度模糊的防御而设计的。...



## **29. Exploiting Edge Features for Transferable Adversarial Attacks in Distributed Machine Learning**

在分布式机器学习中利用边缘特征进行可转移对抗攻击 cs.LG

under review

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.07259v1) [paper-pdf](http://arxiv.org/pdf/2507.07259v1)

**Authors**: Giulio Rossolini, Fabio Brau, Alessandro Biondi, Battista Biggio, Giorgio Buttazzo

**Abstract**: As machine learning models become increasingly deployed across the edge of internet of things environments, a partitioned deep learning paradigm in which models are split across multiple computational nodes introduces a new dimension of security risk. Unlike traditional inference setups, these distributed pipelines span the model computation across heterogeneous nodes and communication layers, thereby exposing a broader attack surface to potential adversaries. Building on these motivations, this work explores a previously overlooked vulnerability: even when both the edge and cloud components of the model are inaccessible (i.e., black-box), an adversary who intercepts the intermediate features transmitted between them can still pose a serious threat. We demonstrate that, under these mild and realistic assumptions, an attacker can craft highly transferable proxy models, making the entire deep learning system significantly more vulnerable to evasion attacks. In particular, the intercepted features can be effectively analyzed and leveraged to distill surrogate models capable of crafting highly transferable adversarial examples against the target model. To this end, we propose an exploitation strategy specifically designed for distributed settings, which involves reconstructing the original tensor shape from vectorized transmitted features using simple statistical analysis, and adapting surrogate architectures accordingly to enable effective feature distillation. A comprehensive and systematic experimental evaluation has been conducted to demonstrate that surrogate models trained with the proposed strategy, i.e., leveraging intermediate features, tremendously improve the transferability of adversarial attacks. These findings underscore the urgent need to account for intermediate feature leakage in the design of secure distributed deep learning systems.

摘要: 随着机器学习模型越来越多地部署在物联网环境的边缘，分区深度学习范式（其中模型在多个计算节点上分裂）引入了新的安全风险维度。与传统的推理设置不同，这些分布式管道跨越异类节点和通信层的模型计算，从而向潜在对手暴露更广泛的攻击面。基于这些动机，这项工作探索了以前被忽视的漏洞：即使模型的边缘和云组件都不可访问（即，黑匣子），拦截它们之间传输的中间特征的对手仍然可能构成严重威胁。我们证明，在这些温和而现实的假设下，攻击者可以制作高度可转移的代理模型，使整个深度学习系统更容易受到规避攻击。特别是，可以有效地分析和利用截获的特征来提取代理模型，这些代理模型能够针对目标模型制作高度可转移的对抗性示例。为此，我们提出了一种专门为分布式设置设计的开发策略，该策略涉及使用简单的统计分析从矢量化传输特征重建原始张量形状，并相应地调整代理架构以实现有效的特征提取。已经进行了全面和系统的实验评估，以证明使用所提出的策略训练的代理模型，即，利用中间功能，极大地提高了对抗性攻击的可转移性。这些发现凸显了在安全分布式深度学习系统的设计中迫切需要考虑中间特征泄漏。



## **30. Protecting Classifiers From Attacks**

保护分类器免受攻击 stat.ML

Published in Statistical Science:  https://projecteuclid.org/journals/statistical-science/volume-39/issue-3/Protecting-Classifiers-from-Attacks/10.1214/24-STS922.full

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2004.08705v2) [paper-pdf](http://arxiv.org/pdf/2004.08705v2)

**Authors**: Victor Gallego, Roi Naveiro, Alberto Redondo, David Rios Insua, Fabrizio Ruggeri

**Abstract**: In multiple domains such as malware detection, automated driving systems, or fraud detection, classification algorithms are susceptible to being attacked by malicious agents willing to perturb the value of instance covariates to pursue certain goals. Such problems pertain to the field of adversarial machine learning and have been mainly dealt with, perhaps implicitly, through game-theoretic ideas with strong underlying common knowledge assumptions. These are not realistic in numerous application domains in relation to security and business competition. We present an alternative Bayesian decision theoretic framework that accounts for the uncertainty about the attacker's behavior using adversarial risk analysis concepts. In doing so, we also present core ideas in adversarial machine learning to a statistical audience. A key ingredient in our framework is the ability to sample from the distribution of originating instances given the, possibly attacked, observed ones. We propose an initial procedure based on approximate Bayesian computation usable during operations; within it, we simulate the attacker's problem taking into account our uncertainty about his elements. Large-scale problems require an alternative scalable approach implementable during the training stage. Globally, we are able to robustify statistical classification algorithms against malicious attacks.

摘要: 在恶意软件检测、自动驾驶系统或欺诈检测等多个领域中，分类算法很容易受到恶意代理的攻击，恶意代理愿意干扰实例协变量的价值以追求某些目标。此类问题涉及对抗性机器学习领域，并且主要通过具有强大基础常识假设的博弈论思想来解决（也许是隐含的）。这些在许多与安全和业务竞争相关的应用程序领域中是不现实的。我们提出了一个替代的Bayesian决策理论框架，该框架使用对抗风险分析概念来解释攻击者行为的不确定性。在此过程中，我们还向统计受众展示了对抗机器学习的核心思想。我们的框架中的一个关键因素是能够从原始实例的分布中进行采样，这些实例可能受到攻击，并被观察到。我们提出了一个初步的过程中可用的近似贝叶斯计算的基础上，在它里面，我们模拟攻击者的问题，考虑到我们的不确定性，他的元素。大规模的问题需要一个替代的可扩展的方法在训练阶段实施。在全球范围内，我们能够增强统计分类算法抵御恶意攻击。



## **31. Robust and Safe Traffic Sign Recognition using N-version with Weighted Voting**

使用加权投票的N版本鲁棒且安全的交通标志识别 cs.LG

27 pages including appendix, 1 figure

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06907v1) [paper-pdf](http://arxiv.org/pdf/2507.06907v1)

**Authors**: Linyun Gao, Qiang Wen, Fumio Machida

**Abstract**: Autonomous driving is rapidly advancing as a key application of machine learning, yet ensuring the safety of these systems remains a critical challenge. Traffic sign recognition, an essential component of autonomous vehicles, is particularly vulnerable to adversarial attacks that can compromise driving safety. In this paper, we propose an N-version machine learning (NVML) framework that integrates a safety-aware weighted soft voting mechanism. Our approach utilizes Failure Mode and Effects Analysis (FMEA) to assess potential safety risks and assign dynamic, safety-aware weights to the ensemble outputs. We evaluate the robustness of three-version NVML systems employing various voting mechanisms against adversarial samples generated using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks. Experimental results demonstrate that our NVML approach significantly enhances the robustness and safety of traffic sign recognition systems under adversarial conditions.

摘要: 自动驾驶作为机器学习的关键应用正在迅速发展，但确保这些系统的安全性仍然是一个严峻的挑战。交通标志识别是自动驾驶汽车的重要组成部分，特别容易受到可能危及驾驶安全的对抗攻击。在本文中，我们提出了一个N版本机器学习（NVML）框架，该框架集成了安全感知加权软投票机制。我们的方法利用故障模式与影响分析（EIA）来评估潜在的安全风险，并为总体输出分配动态的安全意识权重。我们评估了采用各种投票机制的三版本NVML系统针对使用快速梯度符号法（FGSM）和投影梯度下降（PVD）攻击生成的对抗样本的稳健性。实验结果表明，我们的NVML方法显着增强了交通标志识别系统在对抗条件下的鲁棒性和安全性。



## **32. A Single-Point Measurement Framework for Robust Cyber-Attack Diagnosis in Smart Microgrids Using Dual Fractional-Order Feature Analysis**

使用双分数阶特征分析的智能微电网鲁棒网络攻击诊断的单点测量框架 eess.SY

8 pages, 10 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06890v1) [paper-pdf](http://arxiv.org/pdf/2507.06890v1)

**Authors**: Yifan Wang

**Abstract**: Cyber-attacks jeopardize the safe operation of smart microgrids. At the same time, existing diagnostic methods either depend on expensive multi-point instrumentation or stringent modelling assumptions that are untenable under single-sensor constraints. This paper proposes a Fractional-Order Memory-Enhanced Attack-Diagnosis Scheme (FO-MADS) that achieves low-latency fault localisation and cyber-attack detection using only one VPQ (Voltage-Power-Reactive-power) sensor. FO-MADS first constructs a dual fractional-order feature library by jointly applying Caputo and Gr\"unwald-Letnikov derivatives, thereby amplifying micro-perturbations and slow drifts in the VPQ signal. A two-stage hierarchical classifier then pinpoints the affected inverter and isolates the faulty IGBT switch, effectively alleviating class imbalance. Robustness is further strengthened through Progressive Memory-Replay Adversarial Training (PMR-AT), whose attack-aware loss is dynamically re-weighted via Online Hard Example Mining (OHEM) to prioritise the most challenging samples. Experiments on a four-inverter microgrid testbed comprising 1 normal and 24 fault classes under four attack scenarios demonstrate diagnostic accuracies of 96.6 % (bias), 94.0 % (noise), 92.8 % (data replacement), and 95.7 % (replay), while sustaining 96.7 % under attack-free conditions. These results establish FO-MADS as a cost-effective and readily deployable solution that markedly enhances the cyber-physical resilience of smart microgrids.

摘要: 网络攻击危及智能微电网的安全运行。与此同时，现有的诊断方法要么依赖于昂贵的多点仪器，要么依赖于在单传感器约束下站不住脚的严格建模假设。本文提出了一种分数阶存储器增强型攻击诊断方案（FO-MADS），仅使用一个VPQ（电压功率反应功率）传感器即可实现低延迟故障定位和网络攻击检测。FO-MADS首先通过联合应用Caputo和Gr ' unwald-Letnikov衍生物来构建双分数阶特征库，从而放大VPQ信号中的微扰动和缓慢漂移。然后，两级分层分类器确定受影响的逆变器并隔离有故障的绝缘栅双极开关，有效地缓解类别不平衡。通过渐进记忆回放对抗训练（PMR-AT）进一步加强稳健性，其攻击感知损失通过在线硬示例挖掘（OHEEM）动态重新加权，以优先考虑最具挑战性的样本。在四种攻击场景下，在包含1个正常和24个故障类别的四逆变器微电网测试台上进行的实验表明，诊断准确率为96.6%（偏差）、94.0%（噪音）、92.8%（数据替换）和95.7%（重播），而在无攻击条件下保持96.7%。这些结果使FO-MADS成为一种具有成本效益且易于部署的解决方案，可以显着增强智能微电网的网络物理弹性。



## **33. IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization and Perturbation Optimization**

NAP：通过感知本地化和扰动优化进行隐形对抗补丁攻击 cs.CV

Published in ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06856v1) [paper-pdf](http://arxiv.org/pdf/2507.06856v1)

**Authors**: Subrat Kishore Dutta, Xiao Zhang

**Abstract**: Despite modifying only a small localized input region, adversarial patches can drastically change the prediction of computer vision models. However, prior methods either cannot perform satisfactorily under targeted attack scenarios or fail to produce contextually coherent adversarial patches, causing them to be easily noticeable by human examiners and insufficiently stealthy against automatic patch defenses. In this paper, we introduce IAP, a novel attack framework that generates highly invisible adversarial patches based on perceptibility-aware localization and perturbation optimization schemes. Specifically, IAP first searches for a proper location to place the patch by leveraging classwise localization and sensitivity maps, balancing the susceptibility of patch location to both victim model prediction and human visual system, then employs a perceptibility-regularized adversarial loss and a gradient update rule that prioritizes color constancy for optimizing invisible perturbations. Comprehensive experiments across various image benchmarks and model architectures demonstrate that IAP consistently achieves competitive attack success rates in targeted settings with significantly improved patch invisibility compared to existing baselines. In addition to being highly imperceptible to humans, IAP is shown to be stealthy enough to render several state-of-the-art patch defenses ineffective.

摘要: 尽管只修改了很小的局部输入区域，但对抗性补丁可以极大地改变计算机视觉模型的预测。然而，现有方法要么无法在有针对性的攻击场景下令人满意地表现，要么无法产生上下文一致的对抗补丁，导致它们很容易被人类检查员注意到，并且对于自动补丁防御来说不够隐蔽。本文中，我们介绍了NAP，这是一种新型攻击框架，它基于感知定位和扰动优化方案生成高度不可见的对抗补丁。具体来说，TIP首先通过利用类定位和敏感度地图来搜索放置补丁的适当位置，平衡补丁位置对受害者模型预测和人类视觉系统的敏感性，然后采用感知规范化的对抗损失和梯度更新规则，优先考虑颜色稳定性以优化不可见干扰。各种图像基准和模型架构的全面实验表明，与现有基线相比，TIP在目标设置中始终实现有竞争力的攻击成功率，并且补丁不可见性显着提高。除了人类高度难以察觉之外，TIP还被证明足够隐蔽，足以使几种最先进的补丁防御无效。



## **34. PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection**

PBCAT：针对对象检测物理上可实现的攻击的基于补丁的复合对抗训练 cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.23581v2) [paper-pdf](http://arxiv.org/pdf/2506.23581v2)

**Authors**: Xiao Li, Yiming Zhu, Yifan Huang, Wei Zhang, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Object detection plays a crucial role in many security-sensitive applications. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. While AT has been extensively studied in the $l_\infty$ attack settings on classification models, AT against physically realizable attacks on object detectors has received limited exploration. Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored. In this work, we consider defending against various physically realizable attacks with a unified AT method. We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures. Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack.

摘要: 对象检测在许多安全敏感应用程序中发挥着至关重要的作用。然而，最近的几项研究表明，对象检测器很容易被物理上可实现的攻击所愚弄，例如对抗补丁和最近的对抗纹理，这些攻击构成了现实而紧迫的威胁。对抗训练（AT）被认为是对抗攻击的最有效防御。虽然AT在分类模型的$l_\infty$攻击设置中得到了广泛研究，但针对对象检测器物理上可实现的攻击的AT的探索有限。早期的尝试只是为了防御对抗补丁，这使得AT能够对抗更广泛的物理可实现的攻击，但尚未得到充分的探索。在这项工作中，我们考虑使用统一的AT方法来防御各种物理上可实现的攻击。我们提出了PBCAT，这是一种新型的基于补丁的复合对抗训练策略。PBCAT通过结合小区域梯度引导的对抗补丁和覆盖整个图像的不可感知的全局对抗扰动来优化模型。通过这些设计，PBCAT不仅有潜力防御对抗补丁，还有潜力防御不可见的物理可实现的攻击，例如对抗纹理。在多个环境中进行的大量实验表明，与最先进的防御方法相比，PBCAT显着提高了针对各种物理可实现攻击的鲁棒性。值得注意的是，在最近的一次对抗性纹理攻击下，它比之前的防御方法提高了29.7%。



## **35. Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking**

尾部感知对抗攻击：高效LLM越狱的分布式方法 cs.LG

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.04446v2) [paper-pdf](http://arxiv.org/pdf/2507.04446v2)

**Authors**: Tim Beyer, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.

摘要: 为了保证大规模安全、稳健地部署大型语言模型（LLM），准确评估其对抗稳健性至关重要。现有的对抗性攻击通常针对单点贪婪世代的有害响应，忽视了LLM固有的随机性。在本文中，我们提出了一种新颖的对抗稳健性评估框架，该框架对整个输出分布（包括尾部风险）进行显式建模，为模型大规模稳健性提供更好的估计。通过将攻击过程描述为优化和采样之间的资源分配问题，我们确定了计算最优权衡，并表明将采样集成到现有攻击中可将ASB提高高达48%，并将效率提高高达两个数量级。我们的框架还使我们能够分析不同的攻击算法如何影响输出伤害分布。令人惊讶的是，我们发现大多数优化策略对输出危害影响很小。最后，我们引入了一个基于熵最大化的无数据概念验证目标，以演示我们的尾部感知视角如何实现新的优化目标。总体而言，我们的研究结果强调了尾部感知攻击和评估协议对于准确评估和加强LLM安全性的重要性。



## **36. Distributed Fault-Tolerant Multi-Robot Cooperative Localization in Adversarial Environments**

对抗环境下的分布式故障多机器人协同定位 cs.RO

Accepted to IROS 2025 Conference

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06750v1) [paper-pdf](http://arxiv.org/pdf/2507.06750v1)

**Authors**: Tohid Kargar Tasooji, Ramviyas Parasuraman

**Abstract**: In multi-robot systems (MRS), cooperative localization is a crucial task for enhancing system robustness and scalability, especially in GPS-denied or communication-limited environments. However, adversarial attacks, such as sensor manipulation, and communication jamming, pose significant challenges to the performance of traditional localization methods. In this paper, we propose a novel distributed fault-tolerant cooperative localization framework to enhance resilience against sensor and communication disruptions in adversarial environments. We introduce an adaptive event-triggered communication strategy that dynamically adjusts communication thresholds based on real-time sensing and communication quality. This strategy ensures optimal performance even in the presence of sensor degradation or communication failure. Furthermore, we conduct a rigorous analysis of the convergence and stability properties of the proposed algorithm, demonstrating its resilience against bounded adversarial zones and maintaining accurate state estimation. Robotarium-based experiment results show that our proposed algorithm significantly outperforms traditional methods in terms of localization accuracy and communication efficiency, particularly in adversarial settings. Our approach offers improved scalability, reliability, and fault tolerance for MRS, making it suitable for large-scale deployments in real-world, challenging environments.

摘要: 在多机器人系统（MRS）中，协作定位是提高系统鲁棒性和可扩展性的关键任务，特别是在GPS拒绝或通信受限的环境中。然而，对抗性攻击，如传感器操纵和通信干扰，对传统定位方法的性能提出了重大挑战。在本文中，我们提出了一种新的分布式容错合作定位框架，以提高对抗性环境中的传感器和通信中断的弹性。我们介绍了一种自适应事件触发的通信策略，动态调整通信阈值的基础上实时感知和通信质量。即使存在传感器降级或通信故障，该策略也确保了最佳性能。此外，我们对所提出算法的收敛性和稳定性属性进行了严格分析，展示了其对有界对抗区的弹性并保持准确的状态估计。基于机器人馆的实验结果表明，我们提出的算法在定位准确性和通信效率方面显着优于传统方法，特别是在对抗环境中。我们的方法为MR提供了改进的可扩展性、可靠性和耐故障性，使其适合在现实世界、具有挑战性的环境中进行大规模部署。



## **37. Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment**

通过去偏高置信Logit对齐实现对抗鲁棒性 cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2408.06079v2) [paper-pdf](http://arxiv.org/pdf/2408.06079v2)

**Authors**: Kejia Zhang, Juanjuan Weng, Shaozi Li, Zhiming Luo

**Abstract**: Despite the remarkable progress of deep neural networks (DNNs) in various visual tasks, their vulnerability to adversarial examples raises significant security concerns. Recent adversarial training methods leverage inverse adversarial attacks to generate high-confidence examples, aiming to align adversarial distributions with high-confidence class regions. However, our investigation reveals that under inverse adversarial attacks, high-confidence outputs are influenced by biased feature activations, causing models to rely on background features that lack a causal relationship with the labels. This spurious correlation bias leads to overfitting irrelevant background features during adversarial training, thereby degrading the model's robust performance and generalization capabilities. To address this issue, we propose Debiased High-Confidence Adversarial Training (DHAT), a novel approach that aligns adversarial logits with debiased high-confidence logits and restores proper attention by enhancing foreground logit orthogonality. Extensive experiments demonstrate that DHAT achieves state-of-the-art robustness on both CIFAR and ImageNet-1K benchmarks, while significantly improving generalization by mitigating the feature bias inherent in inverse adversarial training approaches. Code is available at https://github.com/KejiaZhang-Robust/DHAT.

摘要: 尽管深度神经网络（DNN）在各种视觉任务中取得了显着进展，但它们对对抗性示例的脆弱性引发了严重的安全问题。最近的对抗训练方法利用反向对抗攻击来生成高置信度示例，旨在将对抗分布与高置信度类别区域保持一致。然而，我们的研究表明，在反向对抗攻击下，高置信度输出受到有偏见的特征激活的影响，导致模型依赖于与标签缺乏因果关系的背景特征。这种虚假的相关偏差导致对抗训练期间过度匹配不相关的背景特征，从而降低模型的稳健性能和概括能力。为了解决这个问题，我们提出了去偏置的高置信对抗训练（DHAT），这是一种新颖的方法，可以将对抗逻辑与去偏置的高置信逻辑对齐，并通过增强前景逻辑的垂直性来恢复适当的注意力。大量实验表明，DHAT在CIFAR和ImageNet-1 K基准测试上都实现了最先进的鲁棒性，同时通过减轻反向对抗训练方法固有的特征偏差来显着提高概括性。代码可在https://github.com/KejiaZhang-Robust/DHAT上获得。



## **38. Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions**

评估和改进大型语言模型的鲁棒性：调查和未来方向 cs.CL

33 pages, 5 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.11111v2) [paper-pdf](http://arxiv.org/pdf/2506.11111v2)

**Authors**: Kun Zhang, Le Wu, Kui Yu, Guangyi Lv, Dacao Zhang

**Abstract**: Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers) to support the community.

摘要: 近年来，大型语言模型（LLM）因其理解和生成自然语言的能力而受到了广泛关注。随着快速发展和广泛应用（例如，代理人，联合情报），LLM的稳健性受到了越来越多的关注。作为许多人工智能应用的核心大脑，LLM的稳健性要求模型不仅要生成一致的内容，还要在处理意外的应用场景（例如，有毒提示、有限的噪音域数据、向外分布（OOD）应用程序等）。在这篇调查论文中，我们对LLM的稳健性进行了彻底的审查，旨在提供该领域的全面概念和方法术语并促进社区发展。具体来说，我们首先给出了LLM稳健性的正式定义，并给出了这篇调查论文的收集协议。然后，根据受干扰的输入类型，我们从以下角度组织本次调查：1）对抗稳健性：解决提示被故意操纵的问题，例如噪音提示、长上下文、数据攻击等; 2）OOD稳健性：处理意想不到的现实世界应用场景，例如OOD检测、零镜头传输、幻觉等; 3）稳健性评估：总结用于验证LLM稳健性的新评估数据集、指标和工具。在从各个角度回顾了代表性作品后，我们讨论并强调了该领域未来的机会和研究方向。同时，我们还组织相关工作并提供易于搜索的项目（https：//github.com/zhangkunzk/Awesome-LLM-Robustness-papers）来支持社区。



## **39. Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning**

图像可以使你的记忆恢复：一种新的针对图像生成模型遗忘的多模式引导攻击 cs.CV

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.07139v1) [paper-pdf](http://arxiv.org/pdf/2507.07139v1)

**Authors**: Renyang Liu, Guanlin Li, Tianwei Zhang, See-Kiong Ng

**Abstract**: Recent advances in image generation models (IGMs), particularly diffusion-based architectures such as Stable Diffusion (SD), have markedly enhanced the quality and diversity of AI-generated visual content. However, their generative capability has also raised significant ethical, legal, and societal concerns, including the potential to produce harmful, misleading, or copyright-infringing content. To mitigate these concerns, machine unlearning (MU) emerges as a promising solution by selectively removing undesirable concepts from pretrained models. Nevertheless, the robustness and effectiveness of existing unlearning techniques remain largely unexplored, particularly in the presence of multi-modal adversarial inputs.   To bridge this gap, we propose Recall, a novel adversarial framework explicitly designed to compromise the robustness of unlearned IGMs. Unlike existing approaches that predominantly rely on adversarial text prompts, Recall exploits the intrinsic multi-modal conditioning capabilities of diffusion models by efficiently optimizing adversarial image prompts with guidance from a single semantically relevant reference image. Extensive experiments across ten state-of-the-art unlearning methods and diverse tasks show that Recall consistently outperforms existing baselines in terms of adversarial effectiveness, computational efficiency, and semantic fidelity with the original textual prompt. These findings reveal critical vulnerabilities in current unlearning mechanisms and underscore the need for more robust solutions to ensure the safety and reliability of generative models. Code and data are publicly available at \textcolor{blue}{https://github.com/ryliu68/RECALL}.

摘要: 图像生成模型（IGM）的最新进展，特别是基于扩散的架构，例如稳定扩散（SD），显着提高了人工智能生成的视觉内容的质量和多样性。然而，它们的生成能力也引发了重大的道德、法律和社会问题，包括产生有害、误导性或侵犯版权内容的可能性。为了缓解这些担忧，机器取消学习（MU）通过选择性地从预训练模型中删除不需要的概念而成为一种有希望的解决方案。然而，现有的去学习技术的稳健性和有效性在很大程度上仍然没有得到探索，特别是在存在多模式对抗输入的情况下。   为了弥合这一差距，我们提出了Recall，这是一种新颖的对抗框架，明确旨在损害未学习的IGM的稳健性。与主要依赖对抗性文本提示的现有方法不同，Recall通过在单个语义相关参考图像的指导下有效优化对抗性图像提示来利用扩散模型固有的多模式条件反射能力。针对十种最先进的学习方法和多样化任务的广泛实验表明，在对抗有效性、计算效率和原始文本提示的语义保真度方面，Recall始终优于现有基线。这些发现揭示了当前取消学习机制中的关键漏洞，并强调了需要更强大的解决方案来确保生成模型的安全性和可靠性。代码和数据可在\textColor{blue}{https：//github.com/ryliu68/RECALL}上公开获取。



## **40. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

21 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2411.08003v2) [paper-pdf](http://arxiv.org/pdf/2411.08003v2)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.

摘要: 在对抗环境中（例如网络攻击和虚假信息攻击）对大型语言模型（LLM）的输出进行归因会带来重大挑战，而且其重要性可能会越来越大。我们从理论和实证的角度来处理这个归因问题，借鉴形式语言理论（极限识别）和对不断扩大的LLM生态系统的数据驱动分析。通过将LLM的一组可能输出建模为形式语言，我们分析有限的文本样本是否可以唯一地确定原始模型。我们的结果表明，在模型之间能力重叠的温和假设下，某些类别的LLM从根本上无法仅从其输出中识别。我们描绘了理论可识别性的四种制度：（1）无限一类确定性（离散）LLM语言不可识别（Gold的经典结果来自1967年）;（2）无限类概率LLM也是不可识别的（通过确定性情况的扩展）;（3）有限类确定性LLM是可识别的（与Angluin的泄密标准一致）;以及（4）即使是有限类的概率LLM也可能是不可识别的（我们提供了一个新的反例来建立这个负结果）。作为对这些理论见解的补充，我们量化了近年来给定输出的合理模型起源（假设空间）数量的爆炸式增长。即使在保守的假设下--每个开源模型最多在一个新厕所上进行微调--不同候选模型的数量也大约每0.5年翻一番，并且允许多数据集微调组合可以产生翻倍的时间短至0.28年。这种组合增长，加上所有模型和潜在用户的暴力可能性归因的非凡计算成本，使得详尽的归因在实践中不可行。



## **41. Dual State-space Fidelity Blade (D-STAB): A Novel Stealthy Cyber-physical Attack Paradigm**

双状态空间富达刀片（D-STAB）：一种新型隐形网络物理攻击范式 eess.SY

accepted by 2025 American Control Conference

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06492v1) [paper-pdf](http://arxiv.org/pdf/2507.06492v1)

**Authors**: Jiajun Shen, Hao Tu, Fengjun Li, Morteza Hashemi, Di Wu, Huazhen Fang

**Abstract**: This paper presents a novel cyber-physical attack paradigm, termed the Dual State-Space Fidelity Blade (D-STAB), which targets the firmware of core cyber-physical components as a new class of attack surfaces. The D-STAB attack exploits the information asymmetry caused by the fidelity gap between high-fidelity and low-fidelity physical models in cyber-physical systems. By designing precise adversarial constraints based on high-fidelity state-space information, the attack induces deviations in high-fidelity states that remain undetected by defenders relying on low-fidelity observations. The effectiveness of D-STAB is demonstrated through a case study in cyber-physical battery systems, specifically in an optimal charging task governed by a Battery Management System (BMS).

摘要: 本文提出了一种新型的网络物理攻击范式，称为双状态空间富达刀片（D-STAB），其目标是核心网络物理组件的硬件，作为一类新型攻击面。D-STAB攻击利用了网络物理系统中高保真度和低保真度物理模型之间的保真度差距造成的信息不对称。通过基于高保真度状态空间信息设计精确的对抗约束，攻击会导致高保真度状态的偏差，而依赖低保真度观察的防御者仍然无法检测到这些偏差。D-STAB的有效性通过网络物理电池系统中的案例研究得到了证明，特别是在由电池管理系统（BMC）管理的最佳充电任务中。



## **42. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

论LLM在对抗性攻击中言语信心的稳健性 cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06489v1) [paper-pdf](http://arxiv.org/pdf/2507.06489v1)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.

摘要: 大型语言模型（LLM）产生的强大言语信心对于部署LLM至关重要，以确保许多高风险应用程序中人机交互的透明度、信任和安全。在本文中，我们首次对对抗攻击下言语信心的稳健性进行了全面研究。我们引入了一个新颖的框架，通过干扰和基于越狱的方法攻击言语信心分数，并表明这些攻击可能会显着危及言语信心估计并导致答案频繁变化。我们研究了各种提示策略、模型大小和应用领域，揭示了当前的信心激发方法很脆弱，并且常用的防御技术在很大程度上无效或适得其反。我们的研究结果强调了迫切需要设计更强大的机制来表达LLM的信心，因为即使是微妙的语义保留修改也可能导致反应中的误导性信心。



## **43. Real AI Agents with Fake Memories: Fatal Context Manipulation Attacks on Web3 Agents**

具有虚假记忆的真实人工智能代理：对Web 3代理的致命上下文操纵攻击 cs.CR

19 pages, 14 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2503.16248v3) [paper-pdf](http://arxiv.org/pdf/2503.16248v3)

**Authors**: Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath

**Abstract**: AI agents integrated with Web3 offer autonomy and openness but raise security concerns as they interact with financial protocols and immutable smart contracts. This paper investigates the vulnerabilities of AI agents within blockchain-based financial ecosystems when exposed to adversarial threats in real-world scenarios. We introduce the concept of context manipulation -- a comprehensive attack vector that exploits unprotected context surfaces, including input channels, memory modules, and external data feeds. It expands on traditional prompt injection and reveals a more stealthy and persistent threat: memory injection. Using ElizaOS, a representative decentralized AI agent framework for automated Web3 operations, we showcase that malicious injections into prompts or historical records can trigger unauthorized asset transfers and protocol violations which could be financially devastating in reality. To quantify these risks, we introduce CrAIBench, a Web3-focused benchmark covering 150+ realistic blockchain tasks. such as token transfers, trading, bridges, and cross-chain interactions, and 500+ attack test cases using context manipulation. Our evaluation results confirm that AI models are significantly more vulnerable to memory injection compared to prompt injection. Finally, we evaluate a comprehensive defense roadmap, finding that prompt-injection defenses and detectors only provide limited protection when stored context is corrupted, whereas fine-tuning-based defenses substantially reduce attack success rates while preserving performance on single-step tasks. These results underscore the urgent need for AI agents that are both secure and fiduciarily responsible in blockchain environments.

摘要: 与Web 3集成的人工智能代理提供了自主性和开放性，但在与金融协议和不可变智能合同交互时会引发安全问题。本文研究了基于区块链的金融生态系统中人工智能代理在现实世界场景中面临对抗威胁时的脆弱性。我们引入了上下文操纵的概念--这是一种全面的攻击载体，可以利用不受保护的上下文表面，包括输入通道、内存模块和外部数据源。它扩展了传统的即时注入，揭示了一种更隐蔽和持久的威胁：记忆注入。使用ElizaOS（用于自动化Web 3操作的代表性去中心化人工智能代理框架），我们展示了恶意注入提示或历史记录可能会引发未经授权的资产转移和协议违规，这在现实中可能会造成经济上的毁灭性。为了量化这些风险，我们引入了CrAIBench，这是一个专注于Web 3的基准测试，涵盖150多个现实的区块链任务。例如代币转移、交易、桥梁和跨链交互，以及500多个使用上下文操纵的攻击测试案例。我们的评估结果证实，与即时注入相比，人工智能模型明显更容易受到记忆注入的影响。最后，我们评估了全面的防御路线图，发现预算注入防御和检测器仅在存储上下文被破坏时提供有限的保护，而基于微调的防御可以大幅降低攻击成功率，同时保留一步任务的性能。这些结果凸显了对区块链环境中既安全又负信托责任的人工智能代理的迫切需求。



## **44. Single Word Change is All You Need: Designing Attacks and Defenses for Text Classifiers**

更改单个单词即可：为文本分类器设计攻击和防御 cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2401.17196v2) [paper-pdf](http://arxiv.org/pdf/2401.17196v2)

**Authors**: Lei Xu, Sarah Alnegheimish, Laure Berti-Equille, Alfredo Cuesta-Infante, Kalyan Veeramachaneni

**Abstract**: In text classification, creating an adversarial example means subtly perturbing a few words in a sentence without changing its meaning, causing it to be misclassified by a classifier. A concerning observation is that a significant portion of adversarial examples generated by existing methods change only one word. This single-word perturbation vulnerability represents a significant weakness in classifiers, which malicious users can exploit to efficiently create a multitude of adversarial examples. This paper studies this problem and makes the following key contributions: (1) We introduce a novel metric \r{ho} to quantitatively assess a classifier's robustness against single-word perturbation. (2) We present the SP-Attack, designed to exploit the single-word perturbation vulnerability, achieving a higher attack success rate, better preserving sentence meaning, while reducing computation costs compared to state-of-the-art adversarial methods. (3) We propose SP-Defense, which aims to improve \r{ho} by applying data augmentation in learning. Experimental results on 4 datasets and BERT and distilBERT classifiers show that SP-Defense improves \r{ho} by 14.6% and 13.9% and decreases the attack success rate of SP-Attack by 30.4% and 21.2% on two classifiers respectively, and decreases the attack success rate of existing attack methods that involve multiple-word perturbations.

摘要: 在文本分类中，创建对抗性示例意味着微妙地扰乱句子中的几个词而不改变其含义，导致其被分类器错误分类。一个令人担忧的观察是，现有方法生成的很大一部分对抗性例子只改变了一个词。这种单字扰动漏洞代表了分类器的一个重大弱点，恶意用户可以利用它有效地创建大量对抗性示例。本文研究了这个问题并做出了以下关键贡献：（1）我们引入了一种新型指标\r{ho}来定量评估分类器对单字扰动的鲁棒性。(2)我们提出了SP-Attack，旨在利用单字扰动漏洞，实现更高的攻击成功率，更好地保留句子含义，同时与最先进的对抗方法相比降低了计算成本。(3)我们提出SP-Defense，旨在通过在学习中应用数据增强来改进\r{ho}。对4个数据集以及BERT和DistilBERT分类器的实验结果表明，SP-Defense在两个分类器上分别提高了14.6%和13.9%，并将SP-Attack的攻击成功率分别降低了30.4%和21.2%，并降低了涉及多字扰动的现有攻击方法的攻击成功率。



## **45. On the Natural Robustness of Vision-Language Models Against Visual Perception Attacks in Autonomous Driving**

自动驾驶中视觉语言模型对视觉感知攻击的自然鲁棒性 cs.CV

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2506.11472v2) [paper-pdf](http://arxiv.org/pdf/2506.11472v2)

**Authors**: Pedram MohajerAnsari, Amir Salarpour, Michael Kühr, Siyu Huang, Mohammad Hamad, Sebastian Steinhorst, Habeeb Olufowobi, Mert D. Pesé

**Abstract**: Autonomous vehicles (AVs) rely on deep neural networks (DNNs) for critical tasks such as traffic sign recognition (TSR), automated lane centering (ALC), and vehicle detection (VD). However, these models are vulnerable to attacks that can cause misclassifications and compromise safety. Traditional defense mechanisms, including adversarial training, often degrade benign accuracy and fail to generalize against unseen attacks. In this work, we introduce Vehicle Vision Language Models (V2LMs), fine-tuned vision-language models specialized for AV perception. Our findings demonstrate that V2LMs inherently exhibit superior robustness against unseen attacks without requiring adversarial training, maintaining significantly higher accuracy than conventional DNNs under adversarial conditions. We evaluate two deployment strategies: Solo Mode, where individual V2LMs handle specific perception tasks, and Tandem Mode, where a single unified V2LM is fine-tuned for multiple tasks simultaneously. Experimental results reveal that DNNs suffer performance drops of 33% to 46% under attacks, whereas V2LMs maintain adversarial accuracy with reductions of less than 8% on average. The Tandem Mode further offers a memory-efficient alternative while achieving comparable robustness to Solo Mode. We also explore integrating V2LMs as parallel components to AV perception to enhance resilience against adversarial threats. Our results suggest that V2LMs offer a promising path toward more secure and resilient AV perception systems.

摘要: 自动驾驶汽车（AV）依赖深度神经网络（DNN）来执行交通标志识别（TSB）、自动车道定中心（ALC）和车辆检测（VD）等关键任务。然而，这些模型很容易受到可能导致错误分类并损害安全性的攻击。传统的防御机制，包括对抗训练，通常会降低良性准确性，并且无法针对不可见的攻击进行概括。在这项工作中，我们介绍了车辆视觉语言模型（V2 LM），这是专门用于AV感知的微调视觉语言模型。我们的研究结果表明，V2 LM本质上对不可见的攻击表现出卓越的鲁棒性，无需对抗训练，在对抗条件下保持比传统DNN显着更高的准确性。我们评估了两种部署策略：Solo模式（单个V2 LM处理特定的感知任务）和Tandem模式（单个统一V2 LM同时针对多个任务进行微调）。实验结果显示，DNN在攻击下性能下降33%至46%，而V2 LM保持对抗准确性，平均下降不到8%。Tandem模式进一步提供了一种内存高效的替代方案，同时实现了与Solo模式相当的稳健性。我们还探索将V2 LM集成为AV感知的并行组件，以增强对抗威胁的弹性。我们的结果表明，V2 LM为更安全和更有弹性的AV感知系统提供了一条有希望的途径。



## **46. The bitter lesson of misuse detection**

误用检测的惨痛教训 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06282v1) [paper-pdf](http://arxiv.org/pdf/2507.06282v1)

**Authors**: Hadrien Mariaccia, Charbel-Raphaël Segerie, Diego Dorn

**Abstract**: Prior work on jailbreak detection has established the importance of adversarial robustness for LLMs but has largely focused on the model ability to resist adversarial inputs and to output safe content, rather than the effectiveness of external supervision systems. The only public and independent benchmark of these guardrails to date evaluates a narrow set of supervisors on limited scenarios. Consequently, no comprehensive public benchmark yet verifies how well supervision systems from the market perform under realistic, diverse attacks. To address this, we introduce BELLS, a Benchmark for the Evaluation of LLM Supervision Systems. The framework is two dimensional: harm severity (benign, borderline, harmful) and adversarial sophistication (direct vs. jailbreak) and provides a rich dataset covering 3 jailbreak families and 11 harm categories. Our evaluations reveal drastic limitations of specialized supervision systems. While they recognize some known jailbreak patterns, their semantic understanding and generalization capabilities are very limited, sometimes with detection rates close to zero when asking a harmful question directly or with a new jailbreak technique such as base64 encoding. Simply asking generalist LLMs if the user question is "harmful or not" largely outperforms these supervisors from the market according to our BELLS score. But frontier LLMs still suffer from metacognitive incoherence, often responding to queries they correctly identify as harmful (up to 30 percent for Claude 3.7 and greater than 50 percent for Mistral Large). These results suggest that simple scaffolding could significantly improve misuse detection robustness, but more research is needed to assess the tradeoffs of such techniques. Our results support the "bitter lesson" of misuse detection: general capabilities of LLMs are necessary to detect a diverse array of misuses and jailbreaks.

摘要: 之前关于越狱检测的工作已经确定了对抗鲁棒性对LLM的重要性，但主要关注的是模型抵抗对抗输入和输出安全内容的能力，而不是外部监督系统的有效性。迄今为止，这些护栏的唯一公开且独立的基准在有限的情况下评估了有限的监管人员。因此，目前还没有全面的公共基准来验证市场监管系统在现实、多样化的攻击下的表现如何。为了解决这个问题，我们引入了BELLS，这是LLM监督系统评估的基准。该框架是二维的：伤害严重性（良性、边缘、有害）和对抗复杂性（直接与越狱），并提供了涵盖3个越狱家庭和11个伤害类别的丰富数据集。我们的评估揭示了专业监督系统的巨大局限性。虽然它们识别了一些已知的越狱模式，但它们的语义理解和概括能力非常有限，有时在直接询问有害问题或使用base 64编码等新的越狱技术时，检测率接近于零。根据我们的BELLS评分，简单地询问多面手LLM用户问题是否“有害”在很大程度上优于这些市场监管人员。但前沿LLM仍然存在元认知不一致的问题，经常对他们正确识别为有害的查询做出回应（Claude 3.7的这一比例高达30%，Mistral Large的这一比例超过50%）。这些结果表明，简单的支架可以显着提高误用检测的鲁棒性，但需要更多的研究来评估此类技术的权衡。我们的结果支持了滥用检测的“惨痛教训”：LLM的通用功能对于检测各种滥用和越狱是必要的。



## **47. ScoreAdv: Score-based Targeted Generation of Natural Adversarial Examples via Diffusion Models**

ScoreAdv：通过扩散模型基于分数的有针对性地生成自然对抗示例 cs.CV

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06078v1) [paper-pdf](http://arxiv.org/pdf/2507.06078v1)

**Authors**: Chihan Huang, Hao Tang

**Abstract**: Despite the success of deep learning across various domains, it remains vulnerable to adversarial attacks. Although many existing adversarial attack methods achieve high success rates, they typically rely on $\ell_{p}$-norm perturbation constraints, which do not align with human perceptual capabilities. Consequently, researchers have shifted their focus toward generating natural, unrestricted adversarial examples (UAEs). GAN-based approaches suffer from inherent limitations, such as poor image quality due to instability and mode collapse. Meanwhile, diffusion models have been employed for UAE generation, but they still rely on iterative PGD perturbation injection, without fully leveraging their central denoising capabilities. In this paper, we introduce a novel approach for generating UAEs based on diffusion models, named ScoreAdv. This method incorporates an interpretable adversarial guidance mechanism to gradually shift the sampling distribution towards the adversarial distribution, while using an interpretable saliency map to inject the visual information of a reference image into the generated samples. Notably, our method is capable of generating an unlimited number of natural adversarial examples and can attack not only classification models but also retrieval models. We conduct extensive experiments on ImageNet and CelebA datasets, validating the performance of ScoreAdv across ten target models in both black-box and white-box settings. Our results demonstrate that ScoreAdv achieves state-of-the-art attack success rates and image quality. Furthermore, the dynamic balance between denoising and adversarial perturbation enables ScoreAdv to remain robust even under defensive measures.

摘要: 尽管深度学习在各个领域取得了成功，但它仍然容易受到对抗攻击。尽管许多现有的对抗攻击方法取得了很高的成功率，但它们通常依赖于$\ell_{p}$-norm扰动约束，这与人类的感知能力不一致。因此，研究人员将重点转向生成自然的、不受限制的对抗性例子（UAE）。基于GAN的方法存在固有的局限性，例如由于不稳定和模式崩溃而导致的图像质量差。与此同时，扩散模型已被用于阿联酋一代，但它们仍然依赖于迭代PVD扰动注入，而没有充分利用其核心去噪能力。本文中，我们介绍了一种基于扩散模型生成UAE的新型方法，名为ScoreAdv。该方法结合了可解释的对抗引导机制，将采样分布逐渐转向对抗分布，同时使用可解释的显着图将参考图像的视觉信息注入到生成的样本中。值得注意的是，我们的方法能够生成无限数量的自然对抗示例，并且不仅可以攻击分类模型，还可以攻击检索模型。我们对ImageNet和CelebA数据集进行了广泛的实验，验证了ScoreAdv在黑盒和白盒设置下在十个目标模型上的性能。我们的结果表明ScoreAdv实现了最先进的攻击成功率和图像质量。此外，去噪和对抗性扰动之间的动态平衡使ScoreAdv即使在防御措施下也能够保持稳健。



## **48. CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations**

CAVGAN：通过对其内部代表的生成性对抗攻击统一LLM的越狱和辩护 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06043v1) [paper-pdf](http://arxiv.org/pdf/2507.06043v1)

**Authors**: Xiaohu Li, Yunfeng Ning, Zepeng Bao, Mayi Xu, Jianhao Chen, Tieyun Qian

**Abstract**: Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN.

摘要: 安全对齐使大型语言模型（LLM）能够获得针对恶意查询的保护，但各种越狱攻击方法揭示了这种安全机制的漏洞。之前的研究已经孤立了LLM越狱攻击和防御。我们分析了LLM的安全保护机制，提出了攻击与防御相结合的框架。我们的方法基于LLM中间层嵌入的线性可分离性质，以及越狱攻击的本质，旨在嵌入有害问题并将其转移到安全区域。我们利用生成对抗网络（GAN）来学习LLM内部的安全判断边界，以实现高效的越狱攻击和防御。实验结果表明，我们的方法在三种流行的LLM中平均越狱成功率为88.85%，而在最先进的越狱数据集上的防御成功率平均达到84.17%。这不仅验证了我们方法的有效性，还揭示了LLM的内部安全机制，为增强模型安全性提供了新的见解。代码和数据可在https://github.com/NLPGM/CAVGAN上获取。



## **49. TuneShield: Mitigating Toxicity in Conversational AI while Fine-tuning on Untrusted Data**

TuneShield：在对不可信数据进行微调的同时减轻对话人工智能中的毒性 cs.CR

Pre-print

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.05660v1) [paper-pdf](http://arxiv.org/pdf/2507.05660v1)

**Authors**: Aravind Cheruvu, Shravya Kanchi, Sifat Muhammad Abdullah, Nicholas Kong, Daphne Yao, Murtuza Jadliwala, Bimal Viswanath

**Abstract**: Recent advances in foundation models, such as LLMs, have revolutionized conversational AI. Chatbots are increasingly being developed by customizing LLMs on specific conversational datasets. However, mitigating toxicity during this customization, especially when dealing with untrusted training data, remains a significant challenge. To address this, we introduce TuneShield, a defense framework designed to mitigate toxicity during chatbot fine-tuning while preserving conversational quality. TuneShield leverages LLM-based toxicity classification, utilizing the instruction-following capabilities and safety alignment of LLMs to effectively identify toxic samples, outperforming industry API services. TuneShield generates synthetic conversation samples, termed 'healing data', based on the identified toxic samples, using them to mitigate toxicity while reinforcing desirable behavior during fine-tuning. It performs an alignment process to further nudge the chatbot towards producing desired responses. Our findings show that TuneShield effectively mitigates toxicity injection attacks while preserving conversational quality, even when the toxicity classifiers are imperfect or biased. TuneShield proves to be resilient against adaptive adversarial and jailbreak attacks. Additionally, TuneShield demonstrates effectiveness in mitigating adaptive toxicity injection attacks during dialog-based learning (DBL).

摘要: 基础模型（如LLM）的最新进展彻底改变了对话式AI。聊天机器人越来越多地通过在特定的会话数据集上定制LLM来开发。然而，在这种定制过程中减轻毒性，特别是在处理不可信的训练数据时，仍然是一个重大挑战。为了解决这个问题，我们引入了TuneShield，这是一个防御框架，旨在减轻聊天机器人微调期间的毒性，同时保持会话质量。TuneShield利用基于LLM的毒性分类，利用LLM的描述跟踪功能和安全性对齐来有效识别有毒样本，优于行业API服务。TuneShield基于识别出的有毒样本生成合成对话样本，称为“治愈数据”，使用它们来减轻毒性，同时在微调期间加强理想的行为。它执行对齐过程，以进一步推动聊天机器人产生所需的响应。我们的研究结果表明，TuneShield可以有效地减轻毒性注入攻击，同时保持对话质量，即使毒性分类器不完美或有偏见。事实证明，TuneShield具有抵御适应性对抗和越狱攻击的能力。此外，TuneShield还证明了在基于对话的学习（DBL）期间减轻适应性毒性注入攻击的有效性。



## **50. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

MEF：一个用于评估黑箱大语言模型脆弱性的能力感知多重加密框架 cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2505.23404v3) [paper-pdf](http://arxiv.org/pdf/2505.23404v3)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have revealed significant vulnerabilities in Large Language Models (LLMs), facilitating the evasion of alignment safeguards through increasingly sophisticated prompt manipulations. In this paper, we propose MEF, a capability-aware multi-encryption framework for evaluating vulnerabilities in black-box LLMs. Our key insight is that the effectiveness of jailbreak strategies can be significantly enhanced by tailoring them to the semantic comprehension capabilities of the target model. We present a typology that classifies LLMs into Type I and Type II based on their comprehension levels, and design adaptive attack strategies for each. MEF combines layered semantic mutations and dual-ended encryption techniques, enabling circumvention of input, inference, and output-level defenses. Experimental results demonstrate the superiority of our approach. Remarkably, it achieves a jailbreak success rate of 98.9\% on GPT-4o (29 May 2025 release). Our findings reveal vulnerabilities in current LLMs' alignment defenses.

摘要: 对抗性越狱攻击的最新进展揭示了大型语言模型（LLM）中的显着漏洞，通过日益复杂的提示操纵促进了对对齐保障措施的规避。在本文中，我们提出了MEF，这是一个用于评估黑匣子LLM中漏洞的功能感知多重加密框架。我们的主要见解是，通过根据目标模型的语义理解能力定制越狱策略，可以显着增强它们的有效性。我们提出了一种类型学，根据它们的理解水平将LLM分为I型和II型，并为每种类型设计自适应攻击策略。MEF结合了分层语义突变和双端加密技术，能够规避输入、推理和输出级防御。实验结果证明了我们方法的优越性。值得注意的是，它在GPT-4 o（2025年5月29日发布）上的越狱成功率达到了98.9%。我们的研究结果揭示了当前LLM对齐防御的漏洞。




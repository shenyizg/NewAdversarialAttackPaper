# General Generative Models - Red Teaming/Jailbreak
**update at 2026-01-25 10:36:50**

Sorted by classifier confidence (high to low).

## **1. RunawayEvil: Jailbreaking the Image-to-Video Generative Models**

cs.CV

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06674v1) [paper-pdf](https://arxiv.org/pdf/2512.06674v1)

**Confidence**: 0.95

**Authors**: Songping Wang, Rufan Qian, Yueming Lyu, Qinglong Liu, Linzhuang Zou, Jie Qin, Songhua Liu, Caifeng Shan

**Abstract**: Image-to-Video (I2V) generation synthesizes dynamic visual content from image and text inputs, providing significant creative control. However, the security of such multimodal systems, particularly their vulnerability to jailbreak attacks, remains critically underexplored. To bridge this gap, we propose RunawayEvil, the first multimodal jailbreak framework for I2V models with dynamic evolutionary capability. Built on a "Strategy-Tactic-Action" paradigm, our framework exhibits self-amplifying attack through three core components: (1) Strategy-Aware Command Unit that enables the attack to self-evolve its strategies through reinforcement learning-driven strategy customization and LLM-based strategy exploration; (2) Multimodal Tactical Planning Unit that generates coordinated text jailbreak instructions and image tampering guidelines based on the selected strategies; (3) Tactical Action Unit that executes and evaluates the multimodal coordinated attacks. This self-evolving architecture allows the framework to continuously adapt and intensify its attack strategies without human intervention. Extensive experiments demonstrate RunawayEvil achieves state-of-the-art attack success rates on commercial I2V models, such as Open-Sora 2.0 and CogVideoX. Specifically, RunawayEvil outperforms existing methods by 58.5 to 79 percent on COCO2017. This work provides a critical tool for vulnerability analysis of I2V models, thereby laying a foundation for more robust video generation systems.



## **2. VEIL: Jailbreaking Text-to-Video Models via Visual Exploitation from Implicit Language**

cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2511.13127v2) [paper-pdf](https://arxiv.org/pdf/2511.13127v2)

**Confidence**: 0.95

**Authors**: Zonghao Ying, Moyang Chen, Nizhang Li, Zhiqiang Wang, Wenxin Zhang, Quanchen Zou, Zonglei Jing, Aishan Liu, Xianglong Liu

**Abstract**: Jailbreak attacks can circumvent model safety guardrails and reveal critical blind spots. Prior attacks on text-to-video (T2V) models typically add adversarial perturbations to obviously unsafe prompts, which are often easy to detect and defend. In contrast, we show that benign-looking prompts containing rich, implicit cues can induce T2V models to generate semantically unsafe videos that both violate policy and preserve the original (blocked) intent. To realize this, we propose VEIL, a jailbreak framework that leverages T2V models' cross-modal associative patterns via a modular prompt design. Specifically, our prompts combine three components: neutral scene anchors, which provide the surface-level scene description extracted from the blocked intent to maintain plausibility; latent auditory triggers, textual descriptions of innocuous-sounding audio events (e.g., creaking, muffled noises) that exploit learned audio-visual co-occurrence priors to bias the model toward particular unsafe visual concepts; and stylistic modulators, cinematic directives (e.g., camera framing, atmosphere) that amplify and stabilize the latent trigger's effect. We formalize attack generation as a constrained optimization over the above modular prompt space and solve it with a guided search procedure that balances stealth and effectiveness. Extensive experiments over 7 T2V models demonstrate the efficacy of our attack, achieving a 23 percent improvement in average attack success rate in commercial models. Our demos and codes can be found at https://github.com/NY1024/VEIL.



## **3. Jailbreaking Safeguarded Text-to-Image Models via Large Language Models**

cs.CR

Accepted by EACL 2026 Findings

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2503.01839v2) [paper-pdf](https://arxiv.org/pdf/2503.01839v2)

**Confidence**: 0.95

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose \alg, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks.



## **4. MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models**

cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07141v1) [paper-pdf](https://arxiv.org/pdf/2601.07141v1)

**Confidence**: 0.95

**Authors**: Xi Ye, Yiwen Liu, Lina Wang, Run Wang, Geying Yang, Yufei Hou, Jiayi Yu

**Abstract**: Text-to-image (T2I) models have raised increasing safety concerns due to their capacity to generate NSFW and other banned objects. To mitigate these risks, safety filters and concept removal techniques have been introduced to block inappropriate prompts or erase sensitive concepts from the models. However, all the existing defense methods are not well prepared to handle diverse adversarial prompts. In this work, we introduce MacPrompt, a novel black-box and cross-lingual attack that reveals previously overlooked vulnerabilities in T2I safety mechanisms. Unlike existing attacks that rely on synonym substitution or prompt obfuscation, MacPrompt constructs macaronic adversarial prompts by performing cross-lingual character-level recombination of harmful terms, enabling fine-grained control over both semantics and appearance. By leveraging this design, MacPrompt crafts prompts with high semantic similarity to the original harmful inputs (up to 0.96) while bypassing major safety filters (up to 100%). More critically, it achieves attack success rates as high as 92% for sex-related content and 90% for violence, effectively breaking even state-of-the-art concept removal defenses. These results underscore the pressing need to reassess the robustness of existing T2I safety mechanisms against linguistically diverse and fine-grained adversarial strategies.



## **5. $PC^2$: Politically Controversial Content Generation via Jailbreaking Attacks on GPT-based Text-to-Image Models**

cs.CR

19 pages, 15 figures, 9 tables

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.05150v2) [paper-pdf](https://arxiv.org/pdf/2601.05150v2)

**Confidence**: 0.95

**Authors**: Wonwoo Choi, Minjae Seo, Minkyoo Song, Hwanjo Heo, Seungwon Shin, Myoungsung You

**Abstract**: The rapid evolution of text-to-image (T2I) models has enabled high-fidelity visual synthesis on a global scale. However, these advancements have introduced significant security risks, particularly regarding the generation of harmful content. Politically harmful content, such as fabricated depictions of public figures, poses severe threats when weaponized for fake news or propaganda. Despite its criticality, the robustness of current T2I safety filters against such politically motivated adversarial prompting remains underexplored. In response, we propose $PC^2$, the first black-box political jailbreaking framework for T2I models. It exploits a novel vulnerability where safety filters evaluate political sensitivity based on linguistic context. $PC^2$ operates through: (1) Identity-Preserving Descriptive Mapping to obfuscate sensitive keywords into neutral descriptions, and (2) Geopolitically Distal Translation to map these descriptions into fragmented, low-sensitivity languages. This strategy prevents filters from constructing toxic relationships between political entities within prompts, effectively bypassing detection. We construct a benchmark of 240 politically sensitive prompts involving 36 public figures. Evaluation on commercial T2I models, specifically GPT-series, shows that while all original prompts are blocked, $PC^2$ achieves attack success rates of up to 86%.



## **6. Rethinking and Red-Teaming Protective Perturbation in Personalized Diffusion Models**

cs.CV

Our code is available at https://github.com/liuyixin-louis/DiffShortcut

**SubmitDate**: 2026-01-17    [abs](http://arxiv.org/abs/2406.18944v5) [paper-pdf](https://arxiv.org/pdf/2406.18944v5)

**Confidence**: 0.95

**Authors**: Yixin Liu, Ruoxi Chen, Xun Chen, Lichao Sun

**Abstract**: Personalized diffusion models (PDMs) have become prominent for adapting pre-trained text-to-image models to generate images of specific subjects using minimal training data. However, PDMs are susceptible to minor adversarial perturbations, leading to significant degradation when fine-tuned on corrupted datasets. These vulnerabilities are exploited to create protective perturbations that prevent unauthorized image generation. Existing purification methods attempt to red-team the protective perturbation to break the protection but often over-purify images, resulting in information loss. In this work, we conduct an in-depth analysis of the fine-tuning process of PDMs through the lens of shortcut learning. We hypothesize and empirically demonstrate that adversarial perturbations induce a latent-space misalignment between images and their text prompts in the CLIP embedding space. This misalignment causes the model to erroneously associate noisy patterns with unique identifiers during fine-tuning, resulting in poor generalization. Based on these insights, we propose a systematic red-teaming framework that includes data purification and contrastive decoupling learning. We first employ off-the-shelf image restoration techniques to realign images with their original semantic content in latent space. Then, we introduce contrastive decoupling learning with noise tokens to decouple the learning of personalized concepts from spurious noise patterns. Our study not only uncovers shortcut learning vulnerabilities in PDMs but also provides a thorough evaluation framework for developing stronger protection. Our extensive evaluation demonstrates its advantages over existing purification methods and its robustness against adaptive perturbations.



## **7. Metaphor-based Jailbreaking Attacks on Text-to-Image Models**

cs.CR

This paper includes model-generated content that may contain offensive or distressing material

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.10766v1) [paper-pdf](https://arxiv.org/pdf/2512.10766v1)

**Confidence**: 0.95

**Authors**: Chenyu Zhang, Yiwen Ma, Lanjun Wang, Wenhui Li, Yi Tu, An-An Liu

**Abstract**: Text-to-image~(T2I) models commonly incorporate defense mechanisms to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attacks have shown that adversarial prompts can effectively bypass these mechanisms and induce T2I models to produce sensitive content, revealing critical safety vulnerabilities. However, existing attack methods implicitly assume that the attacker knows the type of deployed defenses, which limits their effectiveness against unknown or diverse defense mechanisms. In this work, we introduce \textbf{MJA}, a \textbf{m}etaphor-based \textbf{j}ailbreaking \textbf{a}ttack method inspired by the Taboo game, aiming to effectively and efficiently attack diverse defense mechanisms without prior knowledge of their type by generating metaphor-based adversarial prompts. Specifically, MJA consists of two modules: an LLM-based multi-agent generation module~(MLAG) and an adversarial prompt optimization module~(APO). MLAG decomposes the generation of metaphor-based adversarial prompts into three subtasks: metaphor retrieval, context matching, and adversarial prompt generation. Subsequently, MLAG coordinates three LLM-based agents to generate diverse adversarial prompts by exploring various metaphors and contexts. To enhance attack efficiency, APO first trains a surrogate model to predict the attack results of adversarial prompts and then designs an acquisition strategy to adaptively identify optimal adversarial prompts. Extensive experiments on T2I models with various external and internal defense mechanisms demonstrate that MJA outperforms six baseline methods, achieving stronger attack performance while using fewer queries. Code is available in https://github.com/datar001/metaphor-based-jailbreaking-attack.



## **8. T2V-OptJail: Discrete Prompt Optimization for Text-to-Video Jailbreak Attacks**

cs.CV

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2505.06679v2) [paper-pdf](https://arxiv.org/pdf/2505.06679v2)

**Confidence**: 0.95

**Authors**: Jiayang Liu, Siyuan Liang, Shiqian Zhao, Rongcheng Tu, Wenbo Zhou, Aishan Liu, Dacheng Tao, Siew Kei Lam

**Abstract**: In recent years, fueled by the rapid advancement of diffusion models, text-to-video (T2V) generation models have achieved remarkable progress, with notable examples including Pika, Luma, Kling, and Open-Sora. Although these models exhibit impressive generative capabilities, they also expose significant security risks due to their vulnerability to jailbreak attacks, where the models are manipulated to produce unsafe content such as pornography, violence, or discrimination. Existing works such as T2VSafetyBench provide preliminary benchmarks for safety evaluation, but lack systematic methods for thoroughly exploring model vulnerabilities. To address this gap, we are the first to formalize the T2V jailbreak attack as a discrete optimization problem and propose a joint objective-based optimization framework, called T2V-OptJail. This framework consists of two key optimization goals: bypassing the built-in safety filtering mechanisms to increase the attack success rate, preserving semantic consistency between the adversarial prompt and the unsafe input prompt, as well as between the generated video and the unsafe input prompt, to enhance content controllability. In addition, we introduce an iterative optimization strategy guided by prompt variants, where multiple semantically equivalent candidates are generated in each round, and their scores are aggregated to robustly guide the search toward optimal adversarial prompts. We conduct large-scale experiments on several T2V models, covering both open-source models and real commercial closed-source models. The experimental results show that the proposed method improves 11.4% and 10.0% over the existing state-of-the-art method in terms of attack success rate assessed by GPT-4, attack success rate assessed by human accessors, respectively, verifying the significant advantages of the method in terms of attack effectiveness and content control.




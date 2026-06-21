document.addEventListener('DOMContentLoaded', () => {
    const fsInput = document.getElementById('fs');
    const finInput = document.getElementById('fin');
    const ampInput = document.getElementById('amplitude');
    const sampleCountInput = document.getElementById('sample_count');
    const autoCoherent = document.getElementById('auto_coherent');
    const useBinMode = document.getElementById('use_bin_mode');
    const toneBinInput = document.getElementById('tone_bin');
    const toneBinLabel = document.getElementById('tone_bin_label');
    const enableMismatch = document.getElementById('enable_mismatch');
    const referenceMethod = document.getElementById('reference_method');
    const firTapsInput = document.getElementById('fir_taps');
    const firTapsLabel = document.getElementById('fir_taps_label');
    const knownToneSource = document.getElementById('known_tone_source');
    const knownToneSourceLabel = document.getElementById('known_tone_source_label');
    const interMethod = document.getElementById('inter_method');
    const lmsParams = document.getElementById('lms_params');
    const lmsMuInput = document.getElementById('lms_mu');
    const lmsEpochsInput = document.getElementById('lms_epochs');
    const lmsTraceBlockSizeInput = document.getElementById('lms_trace_block_size');
    const lmsSettlingRatioInput = document.getElementById('lms_settling_ratio');

    const noiseSlider = document.getElementById('noise_std_slider');
    const g0Slider = document.getElementById('g0_slider');
    const o0Slider = document.getElementById('o0_slider');
    const g1Slider = document.getElementById('g1_slider');
    const o1Slider = document.getElementById('o1_slider');
    const g2Slider = document.getElementById('g2_slider');
    const o2Slider = document.getElementById('o2_slider');
    const g3Slider = document.getElementById('g3_slider');
    const o3Slider = document.getElementById('o3_slider');
    const dtASlider = document.getElementById('dt_A_slider');
    const dtBSlider = document.getElementById('dt_B_slider');

    const runStatus = document.getElementById('run_status');
    const sndrVal = document.getElementById('sndr_val');
    const enobVal = document.getElementById('enob_val');
    const sfdrVal = document.getElementById('sfdr_val');
    const fftSetupVal = document.getElementById('fft_setup_val');
    const plotImg = document.getElementById('spectrum_plot');
    const loader = document.getElementById('loader');
    const simulateBtn = document.getElementById('simulate_btn');
    const calibrateBtn = document.getElementById('calibrate_btn');
    const mismatchParamsDiv = document.getElementById('mismatch_params');

    const simulationPage = document.getElementById('simulation_page');
    const calibrationPage = document.getElementById('calibration_page');
    const calLoader = document.getElementById('cal_loader');
    const calPlot = document.getElementById('calibration_plot');
    const calibrationMeta = document.getElementById('calibration_meta');
    const calEnobRaw = document.getElementById('cal_enob_raw');
    const calEnobFull = document.getElementById('cal_enob_full');
    const calEnobGain = document.getElementById('cal_enob_gain');
    const calSafeSamples = document.getElementById('cal_safe_samples');
    const performanceTable = document.getElementById('performance_table');
    const coefficientsTable = document.getElementById('coefficients_table');
    const timingTable = document.getElementById('timing_table');
    const diagnosticsTable = document.getElementById('diagnostics_table');
    const outputFiles = document.getElementById('output_files');
    const rawSpectrumPlot = document.getElementById('raw_spectrum_plot');
    const calibratedSpectrumPlot = document.getElementById('calibrated_spectrum_plot');
    const rawSpectrumCaption = document.getElementById('raw_spectrum_caption');
    const calibratedSpectrumCaption = document.getElementById('calibrated_spectrum_caption');

    let lastPayloadKey = null;
    let simulationReady = false;

    const sliderBindings = [
        [noiseSlider, document.getElementById('noise_std_val'), 2],
        [g0Slider, document.getElementById('g0_val'), 3],
        [o0Slider, document.getElementById('o0_val'), 1],
        [g1Slider, document.getElementById('g1_val'), 3],
        [o1Slider, document.getElementById('o1_val'), 1],
        [g2Slider, document.getElementById('g2_val'), 3],
        [o2Slider, document.getElementById('o2_val'), 1],
        [g3Slider, document.getElementById('g3_val'), 3],
        [o3Slider, document.getElementById('o3_val'), 1],
        [dtASlider, document.getElementById('dt_A_val'), 0],
        [dtBSlider, document.getElementById('dt_B_val'), 0],
    ];

    const formatNumber = (value, digits = 2) => {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
            return '--';
        }
        const absValue = Math.abs(numeric);
        if (absValue !== 0 && (absValue < 0.001 || absValue >= 100000)) {
            return numeric.toExponential(digits);
        }
        return numeric.toFixed(digits);
    };

    const formatCell = (value) => {
        if (value === null || value === undefined || value === '-') {
            return '-';
        }
        if (typeof value === 'number') {
            return formatNumber(value, 5).replace(/\.?0+$/, '');
        }
        return String(value);
    };

    const setBusy = (isBusy, label = '运行中...') => {
        simulateBtn.disabled = isBusy;
        calibrateBtn.disabled = isBusy;
        if (isBusy) {
            runStatus.textContent = label;
        }
    };

    const setView = (view) => {
        // No longer toggling views for 3-column layout
    };

    const syncSlider = (slider, display, decimals) => {
        const update = () => {
            display.textContent = Number(slider.value).toFixed(decimals);
        };
        slider.addEventListener('input', update);
        update();
    };

    sliderBindings.forEach(([slider, display, decimals]) => syncSlider(slider, display, decimals));

    const syncMismatchState = () => {
        mismatchParamsDiv.style.opacity = enableMismatch.checked ? '1' : '0.48';
        mismatchParamsDiv.style.pointerEvents = enableMismatch.checked ? 'auto' : 'none';
    };

    const syncFrequencyMode = () => {
        const binMode = useBinMode.checked;
        finInput.disabled = binMode;
        autoCoherent.disabled = binMode;
        toneBinLabel.hidden = !binMode;
    };

    const syncCalibrationOptions = () => {
        const method = referenceMethod.value;
        const useNlms = interMethod.value === 'nlms';
        firTapsInput.disabled = method !== 'fractional_delay_fir';
        knownToneSource.disabled = method !== 'known_tone';
        lmsParams.hidden = !useNlms;
        [lmsMuInput, lmsEpochsInput, lmsTraceBlockSizeInput, lmsSettlingRatioInput]
            .forEach((input) => { input.disabled = !useNlms; });
    };

    enableMismatch.addEventListener('change', syncMismatchState);
    useBinMode.addEventListener('change', syncFrequencyMode);
    referenceMethod.addEventListener('change', syncCalibrationOptions);
    interMethod.addEventListener('change', syncCalibrationOptions);
    syncMismatchState();
    syncFrequencyMode();
    syncCalibrationOptions();

    const collectPayload = () => ({
        fs: Number(fsInput.value) * 1e6,
        fin: Number(finInput.value) * 1e6,
        amplitude: Number(ampInput.value),
        sample_count: parseInt(sampleCountInput.value, 10),
        auto_coherent: autoCoherent.checked,
        frequency_mode: useBinMode.checked ? 'bin' : 'hz',
        tone_bin: parseInt(toneBinInput.value, 10),
        enable_mismatch: enableMismatch.checked,
        noise_std: Number(noiseSlider.value),
        g0: Number(g0Slider.value),
        o0: Number(o0Slider.value),
        g1: Number(g1Slider.value),
        o1: Number(o1Slider.value),
        g2: Number(g2Slider.value),
        o2: Number(o2Slider.value),
        g3: Number(g3Slider.value),
        o3: Number(o3Slider.value),
        dt_mdac_A_ps: Number(dtASlider.value),
        dt_mdac_B_ps: Number(dtBSlider.value),
    });

    const payloadKey = () => JSON.stringify(collectPayload());

    const collectCalibrationPayload = () => {
        let firTaps = parseInt(firTapsInput.value, 10);
        if (!Number.isFinite(firTaps) || firTaps < 3) {
            firTaps = 15;
        }
        if (firTaps % 2 === 0) {
            firTaps += 1;
            firTapsInput.value = String(firTaps);
        }
        const positiveNumber = (input, fallback) => {
            const value = Number(input.value);
            if (!Number.isFinite(value) || value <= 0) {
                input.value = String(fallback);
                return fallback;
            }
            return value;
        };
        const positiveInteger = (input, fallback) => {
            const value = parseInt(input.value, 10);
            if (!Number.isFinite(value) || value < 1) {
                input.value = String(fallback);
                return fallback;
            }
            input.value = String(value);
            return value;
        };
        let settlingRatio = positiveNumber(lmsSettlingRatioInput, 1.1);
        if (settlingRatio < 1) {
            settlingRatio = 1.1;
            lmsSettlingRatioInput.value = String(settlingRatio);
        }
        return {
            reference_method: referenceMethod.value,
            fir_taps: firTaps,
            known_tone_source: knownToneSource.value,
            inter_method: interMethod.value,
            lms_mu: positiveNumber(lmsMuInput, 0.005),
            lms_epochs: positiveInteger(lmsEpochsInput, 1),
            lms_trace_block_size: positiveInteger(lmsTraceBlockSizeInput, 1024),
            lms_settling_ratio: settlingRatio,
        };
    };

    const tableMetricHints = {
        'safe samples': '通过fine范围、参考有效性等筛选后可用于估计/更新的B组样本数。',
        'static samples': '两阶段LS中用于静态gain/offset估计的低斜率样本数；后台NLMS下可能为空。',
        'rank_X': '联合LS设计矩阵的秩，用于判断最小二乘问题是否可辨识；后台NLMS下通常为空。',
        'cond_X': '联合LS设计矩阵条件数，越大表示病态程度越高；后台NLMS下通常为空。',
        'corr_F_s': 'F_B和斜率基准s_B的相关系数，绝对值越大表示gain项和timing-like项越难区分。',
        'rms_res_B_lsb': 'B组校正后相对参考Fhat_B的最终残差RMS，单位LSB；越小表示校准效果越好。',
        'corr_res_F': '最终残差和F_B的相关性，接近0表示gain相关误差被较好消除。',
        'corr_res_s': '最终残差和s_B的相关性，接近0表示timing-like相关误差被较好消除。',
        'adaptive method': '后台自适应算法名称。当前为normalized LMS，即NLMS。',
        'NLMS mu': 'NLMS步长。越大收敛越快但更易抖动或发散，越小越稳但收敛更慢。',
        'NLMS epochs': '对同一批安全样本重复训练的轮数。在线后台校准通常为1轮流式更新。',
        'trace block size': '每隔多少次有效系数更新记录一次RMS和系数轨迹。',
        'settling ratio': '收敛判定比例。阈值为final_rms乘以该比例，例如1.1表示最终RMS的1.1倍以内。',
        'safe samples per epoch': '每轮可用于NLMS更新的安全样本数。',
        'total coefficient updates': '实际执行的NLMS系数更新次数，约等于安全样本数乘以epochs。',
        'convergence reached': '是否找到一个block，使得从该点之后RMS一直保持在settling threshold以内。',
        'convergence block count': '达到收敛判定所需的trace block序号；受trace block size影响。',
        'convergence samples': '达到收敛判定所需的有效NLMS更新样本数；这是观察收敛速度的首选指标，越小越快。',
        'settling threshold RMS': '收敛判定使用的RMS阈值，等于最终rms_res_B_lsb乘以settling ratio。',
        'RMS decay dB/sample': '从第一条trace到最后一条trace的平均RMS下降速度，单位dB/有效更新样本；越大表示下降越快。',
    };

    const escapeAttr = (value) => String(value)
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    const renderTable = (table, mount) => {
        if (!table || !Array.isArray(table.headers) || !Array.isArray(table.rows)) {
            mount.textContent = '--';
            return;
        }
        const renderCell = (cell, columnIndex) => {
            const text = formatCell(cell);
            const hint = columnIndex === 0 ? tableMetricHints[String(cell)] : null;
            const attrs = hint
                ? ` class="hint-cell" title="${escapeAttr(hint)}"`
                : '';
            return `<td${attrs}>${text}</td>`;
        };
        const html = [
            '<table>',
            '<thead><tr>',
            ...table.headers.map((header) => `<th>${formatCell(header)}</th>`),
            '</tr></thead>',
            '<tbody>',
            ...table.rows.map((row) => (
                `<tr>${row.map((cell, columnIndex) => renderCell(cell, columnIndex)).join('')}</tr>`
            )),
            '</tbody></table>',
        ].join('');
        mount.innerHTML = html;
    };

    const renderFiles = (files) => {
        if (!files || typeof files !== 'object') {
            outputFiles.textContent = '--';
            return;
        }
        outputFiles.innerHTML = Object.entries(files)
            .map(([name, path]) => `<div><span>${name}</span><code>${path}</code></div>`)
            .join('');
    };

    const metricCaption = (metrics) => {
        if (!metrics) {
            return '--';
        }
        return `SNDR ${formatNumber(metrics.SNDR_dB, 2)} dB | SFDR ${formatNumber(metrics.SFDR_dBc_single_bin, 2)} dBc | ENOB ${formatNumber(metrics.ENOB_bits, 2)} bit`;
    };

    const describeFrequency = (setup, metrics = {}, fallbackFinHz = null) => {
        const windowName = String(metrics.window || '').replace('4-term ', '');
        const rawBin = setup?.tone_bin;
        const aliasBin = setup?.alias_tone_bin;
        const finHz = Number(setup?.fin_Hz ?? metrics.fin_requested_Hz ?? fallbackFinHz);
        const analysisHz = Number(setup?.analysis_fin_Hz ?? metrics.fin_analysis_Hz ?? metrics.fin_used_Hz ?? fallbackFinHz);
        const finMHz = Number.isFinite(finHz) ? finHz / 1e6 : null;
        const analysisMHz = Number.isFinite(analysisHz) ? analysisHz / 1e6 : finMHz;

        let frequencyText = analysisMHz == null ? '-- MHz' : `${formatNumber(analysisMHz, 6)} MHz`;
        if (Number.isFinite(finMHz) && Number.isFinite(analysisMHz) && Math.abs(finMHz - analysisMHz) > 1e-9) {
            frequencyText = `${formatNumber(finMHz, 6)} MHz in / ${formatNumber(analysisMHz, 6)} MHz FFT`;
        }

        if (rawBin !== undefined && rawBin !== null) {
            const binText = rawBin === aliasBin || aliasBin === undefined || aliasBin === null
                ? `bin ${rawBin}`
                : `bin ${rawBin}->${aliasBin}`;
            return `${binText} | ${frequencyText}${windowName ? ` | ${windowName}` : ''}`;
        }

        return `${frequencyText}${windowName ? ` | ${windowName}` : ''}`;
    };

    const describeCalibrationReference = (config = {}) => {
        const method = config.reference_method || referenceMethod.value;
        if (method === 'fractional_delay_fir') {
            const requested = Number(config.fir_requested_taps ?? config.fir_taps ?? firTapsInput.value);
            const effective = Number(config.fir_effective_taps ?? requested);
            const tapsText = Number.isFinite(effective) && effective !== requested
                ? `taps ${requested}->${effective}`
                : `taps ${Number.isFinite(effective) ? effective : requested}`;
            return `${method} / ${tapsText}`;
        }
        if (method === 'known_tone') {
            return `${method} / source=${config.known_tone_source || knownToneSource.value}`;
        }
        return method;
    };

    const describeInterMethod = (config = {}) => {
        const method = config.inter_method || interMethod.value;
        if (method === 'nlms') {
            const mu = config.lms_mu !== undefined ? ` / mu=${formatNumber(config.lms_mu, 4)}` : '';
            const epochs = config.lms_epochs !== undefined ? ` / epochs=${config.lms_epochs}` : '';
            const block = config.lms_trace_block_size !== undefined ? ` / block=${config.lms_trace_block_size}` : '';
            return `background NLMS${mu}${epochs}${block}`;
        }
        if (method === 'joint') {
            return 'joint LS';
        }
        return 'two-stage LS';
    };

    const updateSimulationView = (data) => {
        sndrVal.textContent = formatNumber(data.metrics.SNDR_dB, 2);
        enobVal.textContent = formatNumber(data.metrics.ENOB_bits, 2);
        sfdrVal.textContent = formatNumber(data.metrics.SFDR_dBc_single_bin, 2);

        fftSetupVal.textContent = `N=${data.metrics.N} | ${describeFrequency(data.fft_setup || {}, data.metrics)}`;

        plotImg.src = `data:image/png;base64,${data.image_b64}`;
        plotImg.hidden = false;
        runStatus.textContent = '仿真完成';
    };

    const runSimulation = async ({ keepView = true } = {}) => {
        const payload = collectPayload();
        setBusy(true, '仿真中...');
        loader.style.display = 'block';
        plotImg.style.opacity = '0.45';

        try {
            const response = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || '仿真接口返回异常');
            }

            const data = await response.json();
            updateSimulationView(data);
            lastPayloadKey = JSON.stringify(payload);
            simulationReady = true;
            if (!keepView) {
                setView('simulation');
            }
            return data;
        } catch (error) {
            console.error('Simulation error:', error);
            runStatus.textContent = '仿真失败';
            window.alert(`仿真失败：${error.message}`);
            throw error;
        } finally {
            setBusy(false);
            simulateBtn.textContent = '运行仿真';
            loader.style.display = 'none';
            plotImg.style.opacity = '1';
        }
    };

    const updateCalibrationView = (data) => {
        const summary = data.summary || {};
        const config = data.config || {};
        calibrationMeta.textContent = `N=${data.sample_count} | ${describeFrequency(data.fft_setup || {}, {}, data.fin_Hz)} | ${describeCalibrationReference(config)} | ${describeInterMethod(config)}`;
        const enobRaw = summary.enob_raw_bits !== undefined ? summary.enob_raw_bits : (summary.sndr_raw_dB != null ? (summary.sndr_raw_dB - 1.76) / 6.02 : null);
        const enobFull = summary.enob_after_full_bits !== undefined ? summary.enob_after_full_bits : (summary.sndr_after_full_dB != null ? (summary.sndr_after_full_dB - 1.76) / 6.02 : null);
        const enobImprov = summary.enob_improvement_bits !== undefined ? summary.enob_improvement_bits : (enobFull != null && enobRaw != null ? enobFull - enobRaw : null);

        calEnobRaw.textContent = formatNumber(enobRaw, 2);
        calEnobFull.textContent = formatNumber(enobFull, 2);
        calEnobGain.textContent = (enobImprov !== undefined && enobImprov !== null && !Number.isNaN(enobImprov)) ? 
            (enobImprov > 0 ? '+' : '') + formatNumber(enobImprov, 2) : '--';
        calSafeSamples.textContent = formatCell(summary.safe_samples);

        calPlot.src = `data:image/png;base64,${data.image_b64}`;
        calPlot.hidden = false;
        rawSpectrumPlot.src = `data:image/png;base64,${data.spectrum_images?.raw || ''}`;
        calibratedSpectrumPlot.src = `data:image/png;base64,${data.spectrum_images?.calibrated || ''}`;
        rawSpectrumPlot.hidden = !data.spectrum_images?.raw;
        calibratedSpectrumPlot.hidden = !data.spectrum_images?.calibrated;
        rawSpectrumCaption.textContent = metricCaption(data.spectrum_metrics?.raw);
        calibratedSpectrumCaption.textContent = metricCaption(data.spectrum_metrics?.calibrated);
        renderTable(data.tables?.performance, performanceTable);
        renderTable(data.tables?.coefficients, coefficientsTable);
        renderTable(data.tables?.timing_mismatch, timingTable);
        renderTable(data.tables?.ls_diagnostics, diagnosticsTable);
        renderFiles(data.files);
        runStatus.textContent = '校准完成';
        setView('calibration');
    };

    const runCalibration = async () => {
        try {
            const currentKey = payloadKey();
            if (!simulationReady || currentKey !== lastPayloadKey) {
                await runSimulation();
            }

            setBusy(true, '校准中...');
            calLoader.style.display = 'block';
            calPlot.style.opacity = '0.45';

            const response = await fetch('/api/calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(collectCalibrationPayload()),
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || '校准接口返回异常');
            }

            const data = await response.json();
            updateCalibrationView(data);
        } catch (error) {
            console.error('Calibration error:', error);
            runStatus.textContent = '校准失败';
            window.alert(`校准失败：${error.message}`);
        } finally {
            setBusy(false);
            calLoader.style.display = 'none';
            calPlot.style.opacity = '1';
        }
    };

    simulateBtn.addEventListener('click', () => runSimulation({ keepView: false }));
    calibrateBtn.addEventListener('click', runCalibration);

    runSimulation();
    setView(window.location.pathname === '/calibration' ? 'calibration' : 'simulation');
});

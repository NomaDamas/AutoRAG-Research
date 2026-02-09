#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');

const SCRIPT_DIR = __dirname;
const SKILL_DIR = path.resolve(SCRIPT_DIR, '..');
const WORKER_PATH = path.join(SCRIPT_DIR, 'council-job-worker.js');

const SKILL_CONFIG_FILE = path.join(SKILL_DIR, 'council.config.yaml');
const REPO_CONFIG_FILE = path.join(path.resolve(SKILL_DIR, '../..'), 'council.config.yaml');

function exitWithError(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function resolveDefaultConfigFile() {
  if (fs.existsSync(SKILL_CONFIG_FILE)) return SKILL_CONFIG_FILE;
  if (fs.existsSync(REPO_CONFIG_FILE)) return REPO_CONFIG_FILE;
  return SKILL_CONFIG_FILE;
}

function detectHostRole() {
  const normalized = SKILL_DIR.replace(/\\/g, '/');
  if (normalized.includes('/.claude/skills/')) return 'claude';
  if (normalized.includes('/.codex/skills/')) return 'codex';
  return 'unknown';
}

function normalizeBool(value) {
  if (value == null) return null;
  const v = String(value).trim().toLowerCase();
  if (['1', 'true', 'yes', 'y', 'on'].includes(v)) return true;
  if (['0', 'false', 'no', 'n', 'off'].includes(v)) return false;
  return null;
}

function resolveAutoRole(role, hostRole) {
  const roleLc = String(role || '').trim().toLowerCase();
  if (roleLc && roleLc !== 'auto') return roleLc;
  if (hostRole === 'codex') return 'codex';
  if (hostRole === 'claude') return 'claude';
  return 'claude';
}

function parseCouncilConfig(configPath) {
  const fallback = {
    council: {
      chairman: { role: 'auto' },
      members: [
        { name: 'claude', command: 'claude -p', emoji: '🧠', color: 'CYAN' },
        { name: 'codex', command: 'codex exec', emoji: '🤖', color: 'BLUE' },
        { name: 'gemini', command: 'gemini', emoji: '💎', color: 'GREEN' },
      ],
      settings: { exclude_chairman_from_members: true, timeout: 120 },
    },
  };

  if (!fs.existsSync(configPath)) return fallback;

  let YAML;
  try {
    YAML = require('yaml');
  } catch {
    exitWithError(
      [
        'Missing runtime dependency: yaml',
        'Your Agent Council installation is out of date.',
        'Reinstall from your project root:',
        '  npx github:team-attention/agent-council --target auto',
      ].join('\n')
    );
  }

  let parsed;
  try {
    parsed = YAML.parse(fs.readFileSync(configPath, 'utf8'));
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    exitWithError(`Invalid YAML in ${configPath}: ${message}`);
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    exitWithError(`Invalid config in ${configPath}: expected a YAML mapping/object at the document root`);
  }
  if (!parsed.council) {
    exitWithError(`Invalid config in ${configPath}: missing required top-level key 'council:'`);
  }
  if (typeof parsed.council !== 'object' || Array.isArray(parsed.council)) {
    exitWithError(`Invalid config in ${configPath}: 'council' must be a mapping/object`);
  }

  const merged = {
    council: {
      chairman: { ...fallback.council.chairman },
      members: Array.isArray(fallback.council.members) ? [...fallback.council.members] : [],
      settings: { ...fallback.council.settings },
    },
  };

  const council = parsed.council;

  if (council.chairman != null) {
    if (typeof council.chairman !== 'object' || Array.isArray(council.chairman)) {
      exitWithError(`Invalid config in ${configPath}: 'council.chairman' must be a mapping/object`);
    }
    merged.council.chairman = { ...merged.council.chairman, ...council.chairman };
  }

  if (Object.prototype.hasOwnProperty.call(council, 'members')) {
    if (!Array.isArray(council.members)) {
      exitWithError(`Invalid config in ${configPath}: 'council.members' must be a list/array`);
    }
    merged.council.members = council.members;
  }

  if (council.settings != null) {
    if (typeof council.settings !== 'object' || Array.isArray(council.settings)) {
      exitWithError(`Invalid config in ${configPath}: 'council.settings' must be a mapping/object`);
    }
    merged.council.settings = { ...merged.council.settings, ...council.settings };
  }

  return merged;
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function safeFileName(name) {
  const cleaned = String(name || '').trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-');
  return cleaned || 'member';
}

function atomicWriteJson(filePath, payload) {
  const tmpPath = `${filePath}.${process.pid}.${crypto.randomBytes(4).toString('hex')}.tmp`;
  fs.writeFileSync(tmpPath, JSON.stringify(payload, null, 2), 'utf8');
  fs.renameSync(tmpPath, filePath);
}

function readJsonIfExists(filePath) {
  try {
    if (!fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function sleepMs(ms) {
  const msNum = Number(ms);
  if (!Number.isFinite(msNum) || msNum <= 0) return;
  const sab = new SharedArrayBuffer(4);
  const view = new Int32Array(sab);
  Atomics.wait(view, 0, 0, Math.trunc(msNum));
}

function computeTerminalDoneCount(counts) {
  const c = counts || {};
  return (
    Number(c.done || 0) +
    Number(c.missing_cli || 0) +
    Number(c.error || 0) +
    Number(c.timed_out || 0) +
    Number(c.canceled || 0)
  );
}

function asCodexStepStatus(value) {
  const v = String(value || '');
  if (v === 'pending' || v === 'in_progress' || v === 'completed') return v;
  return 'pending';
}

function buildCouncilUiPayload(statusPayload) {
  const counts = statusPayload.counts || {};
  const done = computeTerminalDoneCount(counts);
  const total = Number(counts.total || 0);
  const isDone = String(statusPayload.overallState || '') === 'done';

  const queued = Number(counts.queued || 0);
  const running = Number(counts.running || 0);

  const members = Array.isArray(statusPayload.members) ? statusPayload.members : [];
  const sortedMembers = members
    .map((m) => ({
      member: m && m.member != null ? String(m.member) : '',
      state: m && m.state != null ? String(m.state) : 'unknown',
      exitCode: m && m.exitCode != null ? m.exitCode : null,
    }))
    .filter((m) => m.member)
    .sort((a, b) => a.member.localeCompare(b.member));

  const terminalStates = new Set(['done', 'missing_cli', 'error', 'timed_out', 'canceled']);
  // Keep the Plan UI visible by ensuring exactly one `in_progress` item while work remains.
  const dispatchStatus = asCodexStepStatus(isDone ? 'completed' : queued > 0 ? 'in_progress' : 'completed');
  let hasInProgress = dispatchStatus === 'in_progress';

  const memberSteps = sortedMembers.map((m) => {
    const state = m.state || 'unknown';
    const isTerminal = terminalStates.has(state);

    let status;
    if (isTerminal) {
      status = 'completed';
    } else if (!hasInProgress && running > 0 && state === 'running') {
      status = 'in_progress';
      hasInProgress = true;
    } else {
      status = 'pending';
    }

    const label = `[Council] Ask ${m.member}`;
    return { label, status: asCodexStepStatus(status) };
  });

  // Once members are done, the host agent should synthesize and then mark this step completed.
  const synthStatus = asCodexStepStatus(isDone ? (hasInProgress ? 'pending' : 'in_progress') : 'pending');

  const codexPlan = [
    { step: `[Council] Prompt dispatch`, status: dispatchStatus },
    ...memberSteps.map((s) => ({ step: s.label, status: s.status })),
    { step: `[Council] Synthesize`, status: synthStatus },
  ];

  const claudeTodos = [
    {
      content: `[Council] Prompt dispatch`,
      status: dispatchStatus,
      activeForm: dispatchStatus === 'completed' ? 'Dispatched council prompts' : 'Dispatching council prompts',
    },
    ...memberSteps.map((s) => ({
      content: s.label,
      status: s.status,
      activeForm: s.status === 'completed' ? 'Finished' : 'Awaiting response',
    })),
    {
      content: `[Council] Synthesize`,
      status: synthStatus,
      activeForm:
        synthStatus === 'completed'
          ? 'Council results ready'
          : synthStatus === 'in_progress'
            ? 'Ready to synthesize'
            : 'Waiting to synthesize',
    },
  ];

  return {
    progress: { done, total, overallState: String(statusPayload.overallState || '') },
    codex: { update_plan: { plan: codexPlan } },
    claude: { todo_write: { todos: claudeTodos } },
  };
}

function computeStatusPayload(jobDir) {
  const resolvedJobDir = path.resolve(jobDir);
  if (!fs.existsSync(resolvedJobDir)) exitWithError(`jobDir not found: ${resolvedJobDir}`);

  const jobMeta = readJsonIfExists(path.join(resolvedJobDir, 'job.json'));
  if (!jobMeta) exitWithError(`job.json not found: ${path.join(resolvedJobDir, 'job.json')}`);

  const membersRoot = path.join(resolvedJobDir, 'members');
  if (!fs.existsSync(membersRoot)) exitWithError(`members folder not found: ${membersRoot}`);

  const members = [];
  for (const entry of fs.readdirSync(membersRoot)) {
    const statusPath = path.join(membersRoot, entry, 'status.json');
    const status = readJsonIfExists(statusPath);
    if (status) members.push({ safeName: entry, ...status });
  }

  const totals = { queued: 0, running: 0, done: 0, error: 0, missing_cli: 0, timed_out: 0, canceled: 0 };
  for (const m of members) {
    const state = String(m.state || 'unknown');
    if (Object.prototype.hasOwnProperty.call(totals, state)) totals[state]++;
  }

  const allDone = totals.running === 0 && totals.queued === 0;
  const overallState = allDone ? 'done' : totals.running > 0 ? 'running' : 'queued';

  return {
    jobDir: resolvedJobDir,
    id: jobMeta.id || null,
    chairmanRole: jobMeta.chairmanRole || null,
    overallState,
    counts: { total: members.length, ...totals },
    members: members
      .map((m) => ({
        member: m.member,
        state: m.state,
        startedAt: m.startedAt || null,
        finishedAt: m.finishedAt || null,
        exitCode: m.exitCode != null ? m.exitCode : null,
        message: m.message || null,
      }))
      .sort((a, b) => String(a.member).localeCompare(String(b.member))),
  };
}

function parseArgs(argv) {
  const args = argv.slice(2);
  const out = { _: [] };
  const booleanFlags = new Set([
    'json',
    'text',
    'checklist',
    'help',
    'h',
    'verbose',
    'include-chairman',
    'exclude-chairman',
  ]);
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--') {
      out._.push(...args.slice(i + 1));
      break;
    }
    if (!a.startsWith('--')) {
      out._.push(a);
      continue;
    }

    const [key, rawValue] = a.split('=', 2);
    if (rawValue != null) {
      out[key.slice(2)] = rawValue;
      continue;
    }

    const normalizedKey = key.slice(2);
    if (booleanFlags.has(normalizedKey)) {
      out[normalizedKey] = true;
      continue;
    }

    const next = args[i + 1];
    if (next == null || next.startsWith('--')) {
      out[normalizedKey] = true;
      continue;
    }
    out[normalizedKey] = next;
    i++;
  }
  return out;
}

function printHelp() {
  process.stdout.write(`Agent Council (job mode)

Usage:
  council-job.sh start [--config path] [--chairman auto|claude|codex|...] [--jobs-dir path] [--json] "question"
  council-job.sh status [--json|--text|--checklist] [--verbose] <jobDir>
  council-job.sh wait [--cursor CURSOR] [--bucket auto|N] [--interval-ms N] [--timeout-ms N] <jobDir>
  council-job.sh results [--json] <jobDir>
  council-job.sh stop <jobDir>
  council-job.sh clean <jobDir>

Notes:
  - start returns immediately and runs members in parallel via detached Node workers
  - poll status with repeated short calls to update TODO/plan UIs in host agents
  - wait prints JSON by default and blocks until meaningful progress occurs, so you don't spam tool cells
`);
}

function cmdStart(options, prompt) {
  const configPath = options.config || process.env.COUNCIL_CONFIG || resolveDefaultConfigFile();
  const jobsDir =
    options['jobs-dir'] || process.env.COUNCIL_JOBS_DIR || path.join(SKILL_DIR, '.jobs');

  ensureDir(jobsDir);

  const hostRole = detectHostRole();
  const config = parseCouncilConfig(configPath);
  const chairmanRoleRaw = options.chairman || process.env.COUNCIL_CHAIRMAN || config.council.chairman.role || 'auto';
  const chairmanRole = resolveAutoRole(chairmanRoleRaw, hostRole);

  const includeChairman = Boolean(options['include-chairman']);
  const excludeChairmanOverride =
    options['exclude-chairman'] != null ? true : options['include-chairman'] != null ? false : null;

  const excludeSetting = normalizeBool(config.council.settings.exclude_chairman_from_members);
  const excludeChairmanFromMembers =
    excludeChairmanOverride != null ? excludeChairmanOverride : excludeSetting != null ? excludeSetting : true;

  const timeoutSetting = Number(config.council.settings.timeout || 0);
  const timeoutOverride = options.timeout != null ? Number(options.timeout) : null;
  const timeoutSec = Number.isFinite(timeoutOverride) && timeoutOverride > 0 ? timeoutOverride : timeoutSetting > 0 ? timeoutSetting : 0;

  const requestedMembers = config.council.members || [];
  const members = requestedMembers.filter((m) => {
    if (!m || !m.name || !m.command) return false;
    const nameLc = String(m.name).toLowerCase();
    if (excludeChairmanFromMembers && !includeChairman && nameLc === chairmanRole) return false;
    return true;
  });

  const jobId = `${new Date().toISOString().replace(/[:.]/g, '').replace('T', '-').slice(0, 15)}-${crypto
    .randomBytes(3)
    .toString('hex')}`;
  const jobDir = path.join(jobsDir, `council-${jobId}`);
  const membersDir = path.join(jobDir, 'members');
  ensureDir(membersDir);

  fs.writeFileSync(path.join(jobDir, 'prompt.txt'), String(prompt), 'utf8');

  const jobMeta = {
    id: `council-${jobId}`,
    createdAt: new Date().toISOString(),
    configPath,
    hostRole,
    chairmanRole,
    settings: {
      excludeChairmanFromMembers,
      timeoutSec: timeoutSec || null,
    },
    members: members.map((m) => ({
      name: String(m.name),
      command: String(m.command),
      emoji: m.emoji ? String(m.emoji) : null,
      color: m.color ? String(m.color) : null,
    })),
  };
  atomicWriteJson(path.join(jobDir, 'job.json'), jobMeta);

  for (const member of members) {
    const name = String(member.name);
    const safeName = safeFileName(name);
    const memberDir = path.join(membersDir, safeName);
    ensureDir(memberDir);

    atomicWriteJson(path.join(memberDir, 'status.json'), {
      member: name,
      state: 'queued',
      queuedAt: new Date().toISOString(),
      command: String(member.command),
    });

    const workerArgs = [
      WORKER_PATH,
      '--job-dir',
      jobDir,
      '--member',
      name,
      '--safe-member',
      safeName,
      '--command',
      String(member.command),
    ];
    if (timeoutSec && Number.isFinite(timeoutSec) && timeoutSec > 0) {
      workerArgs.push('--timeout', String(timeoutSec));
    }

    const child = spawn(process.execPath, workerArgs, {
      detached: true,
      stdio: 'ignore',
      env: process.env,
    });
    child.unref();
  }

  if (options.json) {
    process.stdout.write(`${JSON.stringify({ jobDir, ...jobMeta }, null, 2)}\n`);
  } else {
    process.stdout.write(`${jobDir}\n`);
  }
}

function cmdStatus(options, jobDir) {
  const payload = computeStatusPayload(jobDir);

  const wantChecklist = Boolean(options.checklist) && !options.json;
  if (wantChecklist) {
    const done = computeTerminalDoneCount(payload.counts);
    const headerId = payload.id ? ` (${payload.id})` : '';
    process.stdout.write(`Agent Council${headerId}\n`);
    process.stdout.write(
      `Progress: ${done}/${payload.counts.total} done  (running ${payload.counts.running}, queued ${payload.counts.queued})\n`
    );
    for (const m of payload.members) {
      const state = String(m.state || '');
      const mark =
        state === 'done'
          ? '[x]'
          : state === 'running' || state === 'queued'
            ? '[ ]'
            : state
              ? '[!]'
              : '[ ]';
      const exitInfo = m.exitCode != null ? ` (exit ${m.exitCode})` : '';
      process.stdout.write(`${mark} ${m.member} — ${state}${exitInfo}\n`);
    }
    return;
  }

  const wantText = Boolean(options.text) && !options.json;
  if (wantText) {
    const done = computeTerminalDoneCount(payload.counts);
    process.stdout.write(`members ${done}/${payload.counts.total} done; running=${payload.counts.running} queued=${payload.counts.queued}\n`);
    if (options.verbose) {
      for (const m of payload.members) {
        process.stdout.write(`- ${m.member}: ${m.state}${m.exitCode != null ? ` (exit ${m.exitCode})` : ''}\n`);
      }
    }
    return;
  }

  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
}

function parseWaitCursor(value) {
  const raw = String(value || '').trim();
  if (!raw) return null;
  const parts = raw.split(':');
  const version = parts[0];
  if (version === 'v1' && parts.length === 4) {
    const bucketSize = Number(parts[1]);
    const doneBucket = Number(parts[2]);
    const isDone = parts[3] === '1';
    if (!Number.isFinite(bucketSize) || bucketSize <= 0) return null;
    if (!Number.isFinite(doneBucket) || doneBucket < 0) return null;
    return { version, bucketSize, dispatchBucket: 0, doneBucket, isDone };
  }
  if (version === 'v2' && parts.length === 5) {
    const bucketSize = Number(parts[1]);
    const dispatchBucket = Number(parts[2]);
    const doneBucket = Number(parts[3]);
    const isDone = parts[4] === '1';
    if (!Number.isFinite(bucketSize) || bucketSize <= 0) return null;
    if (!Number.isFinite(dispatchBucket) || dispatchBucket < 0) return null;
    if (!Number.isFinite(doneBucket) || doneBucket < 0) return null;
    return { version, bucketSize, dispatchBucket, doneBucket, isDone };
  }
  return null;
}

function formatWaitCursor(bucketSize, dispatchBucket, doneBucket, isDone) {
  return `v2:${bucketSize}:${dispatchBucket}:${doneBucket}:${isDone ? 1 : 0}`;
}

function asWaitPayload(statusPayload) {
  const members = Array.isArray(statusPayload.members) ? statusPayload.members : [];
  return {
    jobDir: statusPayload.jobDir,
    id: statusPayload.id,
    chairmanRole: statusPayload.chairmanRole,
    overallState: statusPayload.overallState,
    counts: statusPayload.counts,
    members: members.map((m) => ({
      member: m.member,
      state: m.state,
      exitCode: m.exitCode != null ? m.exitCode : null,
      message: m.message || null,
    })),
    ui: buildCouncilUiPayload(statusPayload),
  };
}

function resolveBucketSize(options, total, prevCursor) {
  const raw = options.bucket != null ? options.bucket : options['bucket-size'];

  if (raw == null || raw === true) {
    if (prevCursor && prevCursor.bucketSize) return prevCursor.bucketSize;
  } else {
    const asString = String(raw).trim().toLowerCase();
    if (asString !== 'auto') {
      const num = Number(asString);
      if (!Number.isFinite(num) || num <= 0) exitWithError(`wait: invalid --bucket: ${raw}`);
      return Math.trunc(num);
    }
  }

  // Auto-bucket: target ~5 updates total.
  const totalNum = Number(total || 0);
  if (!Number.isFinite(totalNum) || totalNum <= 0) return 1;
  return Math.max(1, Math.ceil(totalNum / 5));
}

function cmdWait(options, jobDir) {
  const resolvedJobDir = path.resolve(jobDir);
  const cursorFilePath = path.join(resolvedJobDir, '.wait_cursor');
  const prevCursorRaw =
    options.cursor != null
      ? String(options.cursor)
      : fs.existsSync(cursorFilePath)
        ? String(fs.readFileSync(cursorFilePath, 'utf8')).trim()
        : '';
  const prevCursor = parseWaitCursor(prevCursorRaw);

  const intervalMsRaw = options['interval-ms'] != null ? options['interval-ms'] : 250;
  const intervalMs = Math.max(50, Math.trunc(Number(intervalMsRaw)));
  if (!Number.isFinite(intervalMs) || intervalMs <= 0) exitWithError(`wait: invalid --interval-ms: ${intervalMsRaw}`);

  const timeoutMsRaw = options['timeout-ms'] != null ? options['timeout-ms'] : 0;
  const timeoutMs = Math.trunc(Number(timeoutMsRaw));
  if (!Number.isFinite(timeoutMs) || timeoutMs < 0) exitWithError(`wait: invalid --timeout-ms: ${timeoutMsRaw}`);

  // Always read once to decide bucket sizing and (when no cursor is given) return immediately.
  let payload = computeStatusPayload(jobDir);
  const bucketSize = resolveBucketSize(options, payload.counts.total, prevCursor);

  const doneCount = computeTerminalDoneCount(payload.counts);
  const isDone = payload.overallState === 'done';
  const total = Number(payload.counts.total || 0);
  const queued = Number(payload.counts.queued || 0);
  const dispatchBucket = queued === 0 && total > 0 ? 1 : 0;
  const doneBucket = Math.floor(doneCount / bucketSize);
  const cursor = formatWaitCursor(bucketSize, dispatchBucket, doneBucket, isDone);

  if (!prevCursor) {
    fs.writeFileSync(cursorFilePath, cursor, 'utf8');
    process.stdout.write(`${JSON.stringify({ ...asWaitPayload(payload), cursor }, null, 2)}\n`);
    return;
  }

  const start = Date.now();
  while (cursor === prevCursorRaw) {
    if (timeoutMs > 0 && Date.now() - start >= timeoutMs) break;
    sleepMs(intervalMs);
    payload = computeStatusPayload(jobDir);
    const d = computeTerminalDoneCount(payload.counts);
    const doneFlag = payload.overallState === 'done';
    const totalCount = Number(payload.counts.total || 0);
    const queuedCount = Number(payload.counts.queued || 0);
    const dispatchB = queuedCount === 0 && totalCount > 0 ? 1 : 0;
    const doneB = Math.floor(d / bucketSize);
    const nextCursor = formatWaitCursor(bucketSize, dispatchB, doneB, doneFlag);
    if (nextCursor !== prevCursorRaw) {
      fs.writeFileSync(cursorFilePath, nextCursor, 'utf8');
      process.stdout.write(`${JSON.stringify({ ...asWaitPayload(payload), cursor: nextCursor }, null, 2)}\n`);
      return;
    }
  }

  // Timeout: return current state (cursor may be unchanged).
  const finalPayload = computeStatusPayload(jobDir);
  const finalDone = computeTerminalDoneCount(finalPayload.counts);
  const finalDoneFlag = finalPayload.overallState === 'done';
  const finalTotal = Number(finalPayload.counts.total || 0);
  const finalQueued = Number(finalPayload.counts.queued || 0);
  const finalDispatchBucket = finalQueued === 0 && finalTotal > 0 ? 1 : 0;
  const finalDoneBucket = Math.floor(finalDone / bucketSize);
  const finalCursor = formatWaitCursor(bucketSize, finalDispatchBucket, finalDoneBucket, finalDoneFlag);
  fs.writeFileSync(cursorFilePath, finalCursor, 'utf8');
  process.stdout.write(`${JSON.stringify({ ...asWaitPayload(finalPayload), cursor: finalCursor }, null, 2)}\n`);
}

function cmdResults(options, jobDir) {
  const resolvedJobDir = path.resolve(jobDir);
  const jobMeta = readJsonIfExists(path.join(resolvedJobDir, 'job.json'));
  const membersRoot = path.join(resolvedJobDir, 'members');

  const members = [];
  if (fs.existsSync(membersRoot)) {
    for (const entry of fs.readdirSync(membersRoot)) {
      const statusPath = path.join(membersRoot, entry, 'status.json');
      const outputPath = path.join(membersRoot, entry, 'output.txt');
      const errorPath = path.join(membersRoot, entry, 'error.txt');
      const status = readJsonIfExists(statusPath);
      if (!status) continue;
      const output = fs.existsSync(outputPath) ? fs.readFileSync(outputPath, 'utf8') : '';
      const stderr = fs.existsSync(errorPath) ? fs.readFileSync(errorPath, 'utf8') : '';
      members.push({ safeName: entry, ...status, output, stderr });
    }
  }

  if (options.json) {
    process.stdout.write(
      `${JSON.stringify(
        {
          jobDir: resolvedJobDir,
          id: jobMeta ? jobMeta.id : null,
          prompt: fs.existsSync(path.join(resolvedJobDir, 'prompt.txt'))
            ? fs.readFileSync(path.join(resolvedJobDir, 'prompt.txt'), 'utf8')
            : null,
          members: members
            .map((m) => ({
              member: m.member,
              state: m.state,
              exitCode: m.exitCode != null ? m.exitCode : null,
              message: m.message || null,
              output: m.output,
              stderr: m.stderr,
            }))
            .sort((a, b) => String(a.member).localeCompare(String(b.member))),
        },
        null,
        2
      )}\n`
    );
    return;
  }

  for (const m of members.sort((a, b) => String(a.member).localeCompare(String(b.member)))) {
    process.stdout.write(`\n=== ${m.member} (${m.state}) ===\n`);
    if (m.message) process.stdout.write(`${m.message}\n`);
    process.stdout.write(m.output || '');
    if (!m.output && m.stderr) {
      process.stdout.write('\n');
      process.stdout.write(m.stderr);
    }
    process.stdout.write('\n');
  }
}

function cmdStop(_options, jobDir) {
  const resolvedJobDir = path.resolve(jobDir);
  const membersRoot = path.join(resolvedJobDir, 'members');
  if (!fs.existsSync(membersRoot)) exitWithError(`No members folder found: ${membersRoot}`);

  let stoppedAny = false;
  for (const entry of fs.readdirSync(membersRoot)) {
    const statusPath = path.join(membersRoot, entry, 'status.json');
    const status = readJsonIfExists(statusPath);
    if (!status) continue;
    if (status.state !== 'running') continue;
    if (!status.pid) continue;

    try {
      process.kill(Number(status.pid), 'SIGTERM');
      stoppedAny = true;
    } catch {
      // ignore
    }
  }

  process.stdout.write(stoppedAny ? 'stop: sent SIGTERM to running members\n' : 'stop: no running members\n');
}

function cmdClean(_options, jobDir) {
  const resolvedJobDir = path.resolve(jobDir);
  fs.rmSync(resolvedJobDir, { recursive: true, force: true });
  process.stdout.write(`cleaned: ${resolvedJobDir}\n`);
}

function main() {
  const options = parseArgs(process.argv);
  const [command, ...rest] = options._;

  if (!command || options.help || options.h) {
    printHelp();
    return;
  }

  if (command === 'start') {
    const prompt = rest.join(' ').trim();
    if (!prompt) exitWithError('start: missing prompt');
    cmdStart(options, prompt);
    return;
  }
  if (command === 'status') {
    const jobDir = rest[0];
    if (!jobDir) exitWithError('status: missing jobDir');
    cmdStatus(options, jobDir);
    return;
  }
  if (command === 'wait') {
    const jobDir = rest[0];
    if (!jobDir) exitWithError('wait: missing jobDir');
    cmdWait(options, jobDir);
    return;
  }
  if (command === 'results') {
    const jobDir = rest[0];
    if (!jobDir) exitWithError('results: missing jobDir');
    cmdResults(options, jobDir);
    return;
  }
  if (command === 'stop') {
    const jobDir = rest[0];
    if (!jobDir) exitWithError('stop: missing jobDir');
    cmdStop(options, jobDir);
    return;
  }
  if (command === 'clean') {
    const jobDir = rest[0];
    if (!jobDir) exitWithError('clean: missing jobDir');
    cmdClean(options, jobDir);
    return;
  }

  exitWithError(`Unknown command: ${command}`);
}

if (require.main === module) {
  main();
}

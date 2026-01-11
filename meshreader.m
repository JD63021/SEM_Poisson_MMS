function [nodeInfo, elemInfo, boundaryInfo] = meshreader(gmshFile, order, excludedFlags)
% meshreader  Read a Gmsh .m mesh (struct 'msh' or raw arrays) with quad/line entities
% and return generic node/element/boundary info for Qp quads.
%
% USAGE
%   [nodes, elems, bcs] = meshreader('squaretest10.m', 10);
%   [nodes, elems, bcs] = meshreader('box_q3.m', 3, [99]);  % exclude flag 99
%
% INPUT
%   gmshFile     : string; MATLAB file that defines struct 'msh' (or raw arrays)
%   order        : polynomial order p (>=1)
%   excludedFlags: optional vector of boundary flags to skip (default [])
%
% OUTPUT
%   nodeInfo: struct
%       .id  (NNx1) node ids (1..NN)
%       .x   (NNx1)
%       .y   (NNx1)
%       .N   (=NN)
%
%   elemInfo: struct
%       .quads        (NE x nloc)  connectivity (nloc=(p+1)^2), **Gmsh local order**
%       .quadsQ1      (NE x 4)     corner connectivity (columns 1:4)
%       .order        (=p)
%       .nloc         (=(p+1)^2)
%       .NE           (= #elements)
%
%   boundaryInfo: struct
%       .lines2/.lines3/.../.linesK   (structs with fields flag_<id> = (Nk x K))
%       .allNodes                      (unique boundary node ids after excluding flags)
%
% NOTES
% * Last column of LINES*/QUADS* is assumed to be a physical flag/tag (and ignored in connectivity).
% * Elements are returned exactly in the order given by Gmsh (no reordering).
% * Corner nodes (quadsQ1) are columns 1:4 of the QUADS* array (Gmsh default).

    if nargin < 3, excludedFlags = []; end
    validateattributes(order, {'numeric'},{'scalar','integer','>=',1}, mfilename, 'order', 2);

    % -- Load 'msh' or raw arrays from the .m file
    S = load_gmsh_m(gmshFile);

    % -- Nodes
    POS = mustfield(S, 'POS', 'POS missing in mesh.');
    POS = double(POS);
    nodeInfo.id = (1:size(POS,1)).';
    nodeInfo.x  = POS(:,1);
    nodeInfo.y  = POS(:,2);
    nodeInfo.N  = size(POS,1);

    % -- Elements for given order p
    nloc  = (order+1)^2;
    quads = pick_quads_field(S, order, nloc);          % robust lookup
    assert(size(quads,2) >= nloc, 'QUADS* field has fewer than %d columns.', nloc);

    elem_conn = double(quads(:, 1:nloc));              % drop trailing tag cols
    elemInfo.order  = order;
    elemInfo.nloc   = nloc;
    elemInfo.quads  = elem_conn;
    elemInfo.NE     = size(elem_conn,1);
    elemInfo.quadsQ1 = elem_conn(:,1:4);               % corners are columns 1:4

    % -- Boundaries: collect all LINES* fields present (any K)
    boundaryInfo = struct();
    allB = [];

    lineFields = find_line_fields(S);   % e.g., {'LINES','LINES3','LINES4','LINES11'}
    for i = 1:numel(lineFields)
        fname = lineFields{i};
        raw = double(S.(fname));
        if isempty(raw), continue; end

        K = parse_lines_arity(fname);           % nodes per line element
        if size(raw,2) < K
            warning('Field %s has fewer than %d columns; skipping.', fname, K);
            continue;
        end

        % Last column is assumed to be the flag/tag
        flags = raw(:, end);
        conn  = raw(:, 1:K);

        % Exclude flags if requested
        if ~isempty(excludedFlags)
            keep  = ~ismember(flags, excludedFlags);
            flags = flags(keep);
            conn  = conn(keep, :);
        end

        if isempty(conn), continue; end

        allB = [allB; conn]; %#ok<AGROW>

        % Store per-flag arrays
        group = struct();
        uflags = unique(flags);
        for f = reshape(uflags,1,[])
            mask = (flags == f);
            group.(sprintf('flag_%d', f)) = conn(mask, :);
        end
        boundaryInfo.(sprintf('lines%d', K)) = group;
    end

    boundaryInfo.allNodes = unique(allB(:));

end

% ================= helpers =================

function S = load_gmsh_m(gmshFile)
    % Execute the .m file in its own workspace and capture variables
    run(gmshFile);
    if exist('msh','var') == 1 && isstruct(msh)
        S = harvest_numeric_fields(msh);
        % Some exporters also put POS at top-level
        if ~isfield(S,'POS') && exist('POS','var'), S.POS = POS; end
    else
        % No 'msh' struct → harvest any top-level numeric arrays (POS, QUADS*, LINES*, …)
        ws = whos;
        S  = struct();
        for k = 1:numel(ws)
            name = ws(k).name;
            val  = eval(name); %#ok<EVLDIR>
            if isnumeric(val), S.(name) = val; end
        end
    end
end

function T = harvest_numeric_fields(msh)
    f = fieldnames(msh);
    T = struct();
    for k = 1:numel(f)
        v = msh.(f{k});
        if isnumeric(v), T.(f{k}) = v; end
    end
end

function A = mustfield(S, name, msg)
    assert(isfield(S, name), msg);
    A = S.(name);
end

function quads = pick_quads_field(S, p, nloc)
% Robustly pick the appropriate QUADS* field:
%  - For p=1: prefer QUADS4, else QUADS
%  - For p>1: prefer exact QUADS<nloc>, else best candidate among all fields
%    starting with 'QUADS' that have >= nloc columns.
    if p == 1
        if isfield(S,'QUADS4'), quads = S.QUADS4; return; end
        if isfield(S,'QUADS'),  quads = S.QUADS;  return; end
        % Some exporters still write QUADS4 even when also having QUADS present,
        % but if neither exists, try to find any QUADS* with >=4 columns.
        [ok, quads] = best_quads_candidate(S, 4);
        if ok, return; end
        error('No QUADS/QUADS4 found for p=1.');
    else
        fname = sprintf('QUADS%d', nloc);
        if isfield(S, fname), quads = S.(fname); return; end
        % Fall back: any QUADS* that has >= nloc columns
        [ok, quads] = best_quads_candidate(S, nloc);
        if ok, return; end
        error('No %s found for p=%d, and no suitable QUADS* fallback.', fname, p);
    end
end

function [ok, best] = best_quads_candidate(S, nloc)
    ok = false; best = [];
    f = fieldnames(S);
    % Find every field that looks like QUADS or QUADS<number>
    cand = struct('name',{},'A',{},'cols',{});
    for k = 1:numel(f)
        nm = f{k};
        up = upper(nm);
        if strcmp(up,'QUADS') || startsWith(up,'QUADS')
            A = S.(nm);
            if isnumeric(A) && ~isempty(A) && size(A,2) >= nloc
                cand(end+1).name = nm; %#ok<AGROW>
                cand(end).A      = A;
                cand(end).cols   = size(A,2);
            end
        end
    end
    if isempty(cand), return; end
    % Prefer exact match in column count if available, else the smallest cols >= nloc
    exact = find([cand.cols] == nloc, 1);
    if ~isempty(exact)
        best = cand(exact).A; ok = true; return;
    end
    [~, idx] = min([cand.cols]);        % take the tightest fit
    best = cand(idx).A; ok = true;
end

function lineFields = find_line_fields(S)
    % Return all field names that are 'LINES' or 'LINES<number>'
    f = fieldnames(S);
    lineFields = {};
    for i=1:numel(f)
        nm = f{i}; up = upper(nm);
        if strcmp(up,'LINES') || startsWith(up,'LINES')
            if isnumeric(S.(nm))
                lineFields{end+1} = nm; %#ok<AGROW>
            end
        end
    end
end

function K = parse_lines_arity(fname)
    % 'LINES' -> 2, 'LINES3' -> 3, 'LINES11' -> 11, etc.
    up = upper(fname);
    if strcmp(up,'LINES')
        K = 2;
        return;
    end
    tok = regexp(up,'^LINES(\d+)$','tokens','once');
    if isempty(tok)
        K = 2;
    else
        K = max(2, str2double(tok{1}));
    end
end

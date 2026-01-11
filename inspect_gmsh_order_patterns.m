function inspect_gmsh_order_patterns()
% For each mesh, infer p, recover Gmsh->tensor mapping per element,
% classify vs the 8 D4 symmetries, and print counts.

    files = arrayfun(@(k) sprintf('squaretest%d.m',k), 1:10, 'uni', 0);

    for f = 1:numel(files)
        meshFile = files{f};
        try
            msh = load_gmsh_m(meshFile);
        catch ME
            fprintf('--- %s : could not load (%s)\n', meshFile, ME.message);
            continue;
        end

        % choose highest-order QUADS* field; infer p
        [E, p] = pick_quads_and_order(msh);
        if isempty(E)
            fprintf('--- %s : no usable QUADS* field found.\n', meshFile);
            continue;
        end

        NE   = size(E,1);
        Nloc = size(E,2);
        N1   = p+1;

        IJ = ij_table(N1);                 % tensor enumeration (i fast, j slow)
        r1 = linspace(0,1,N1).';           % 1D equispaced nodes
        w1 = bary_weights(r1);

        % reference element (elem 1)
        ids1 = E(1,:).';
        XY1  = msh.POS(ids1,1:2);
        [dx1,dy1,xmin1,ymin1] = bbox(XY1);
        xi1  = (XY1(:,1)-xmin1)/dx1;
        eta1 = (XY1(:,2)-ymin1)/dy1;
        g2t_ref = gmsh_to_tensor_perm(xi1, eta1, r1, w1, IJ);   % gmsh->tensor

        % D4 transforms on tensor indices
        T = d4_transforms(N1, IJ);

        counts = zeros(8,1);
        per_elem_label = zeros(NE,1);

        for e = 1:NE
            ids = E(e,:).';
            XY  = msh.POS(ids,1:2);
            [dx,dy,xmin,ymin] = bbox(XY);
            xi   = (XY(:,1)-xmin)/dx;
            eta  = (XY(:,2)-ymin)/dy;
            g2t  = gmsh_to_tensor_perm(xi, eta, r1, w1, IJ);

            label = classify_d4(g2t_ref, g2t, T);  % 1..8
            per_elem_label(e) = label;
            counts(label) = counts(label) + 1;
        end

        labels = ["R0","R90","R180","R270","RefX","RefY","Swap","Swap+R180"];
        fprintf('--- %s : p=%d (Nloc=%d, NE=%d) ---\n', meshFile, p, Nloc, NE);
        fprintf('D4 counts relative to reference(elem1):\n');
        for k = 1:8
            fprintf('  %-9s: %d\n', labels(k), counts(k));
        end
        if all(per_elem_label == per_elem_label(1))
            fprintf('All elements share the same pattern: %s\n\n', labels(per_elem_label(1)));
        else
            fprintf('Mixed patterns present.\n\n');
        end
    end
end

% ================ helpers ================

function msh = load_gmsh_m(gmshFile)
    run(gmshFile);  % must define 'msh' or raw arrays
    if exist('msh','var'), S = msh; else, S = struct(); end
    if ~isfield(S,'POS') && exist('POS','var'), S.POS = POS; end
    assert(isfield(S,'POS'), 'POS missing in mesh.');
    S.POS = double(S.POS);
    fns = fieldnames(S);
    for k = 1:numel(fns)
        fn = fns{k};
        if startsWith(fn,'QUADS','IgnoreCase',true) || startsWith(fn,'LINES','IgnoreCase',true)
            S.(fn) = double(S.(fn));
        end
    end
    msh = S;
end

function [E, p] = pick_quads_and_order(msh)
% Pick a QUADS* with columns matching (p+1)^2 (or one more for flag).
    E = []; p = [];
    fns = fieldnames(msh);
    best_p = -inf; best_E = [];
    for k = 1:numel(fns)
        fn = fns{k};
        if ~startsWith(fn,'QUADS','IgnoreCase',true), continue; end
        A = msh.(fn); if isempty(A), continue; end
        ncol = size(A,2);
        for nc = unique([ncol, max(0,ncol-1)])
            if nc < 4, continue; end
            ptry = sqrt(nc) - 1;
            if abs(ptry - round(ptry)) < 1e-12
                ptry = round(ptry);
                Etry = A(:,1:nc);
                if max(Etry,[],'all') <= size(msh.POS,1) && min(Etry,[],'all') >= 1
                    if ptry > best_p
                        best_p = ptry; best_E = Etry;
                    end
                end
            end
        end
    end
    if best_p > 0, E = best_E; p = best_p; end
end

function IJ = ij_table(N1)
    N2 = N1*N1; IJ = zeros(N2,2); c=0;
    for j=1:N1, for i=1:N1, c=c+1; IJ(c,:)=[i,j]; end, end
end

function [dx,dy,xmin,ymin] = bbox(XY)
    xmin = min(XY(:,1)); xmax = max(XY(:,1)); dx = xmax - xmin;
    ymin = min(XY(:,2)); ymax = max(XY(:,2)); dy = ymax - ymin;
    if dx==0, dx=1; end
    if dy==0, dy=1; end
end

function w = bary_weights(x)
    x=x(:); n=numel(x); w=ones(n,1);
    for j=1:n, for k=[1:j-1,j+1:n], w(j)=w(j)*(x(j)-x(k)); end, end
    w=1./w;
end

function L = bary_eval_matrix(xnodes,w,xq)
    xn=xnodes(:); xq=xq(:).'; L=zeros(numel(xn),numel(xq));
    for k=1:numel(xq)
        x=xq(k); d=x-xn; hit=find(abs(d)<1e-14,1);
        if ~isempty(hit), e=zeros(numel(xn),1); e(hit)=1; L(:,k)=e;
        else, t=w./d; L(:,k)=t/sum(t);
        end
    end
end

function g2t = gmsh_to_tensor_perm(xi_nodes, eta_nodes, r1, w1, IJ)
    Nloc = numel(xi_nodes);
    N1   = numel(r1);
    Lx = bary_eval_matrix(r1,w1,xi_nodes.');
    Ly = bary_eval_matrix(r1,w1,eta_nodes.');
    V = zeros(Nloc,Nloc);
    for m=1:Nloc
        i = IJ(m,1); j = IJ(m,2);
        V(:,m) = (Lx(i,:).').*(Ly(j,:).');
    end
    [~,rows] = max(V,[],1);     % rows(m) = gmsh row where tensor m peaks
    g2t = zeros(Nloc,1);
    for m=1:Nloc, g=rows(m); g2t(g)=m; end
end

function T = d4_transforms(N1, IJ)
% Build 8 transforms on tensor indices; returns N2×1 vectors of m' indices.
    N2 = N1*N1;
    % index from [i',j'] vector
    idx = @(IJp) (IJp(2)-1)*N1 + IJp(1);
    % apply a 2D transform F(i,j) -> [i',j'] across all tensor indices
    map = @(F) arrayfun(@(m) idx(F(IJ(m,1), IJ(m,2))), (1:N2).');

    R0   = @(i,j) [ i,           j          ];
    R90  = @(i,j) [ N1-j+1,      i          ];
    R180 = @(i,j) [ N1-i+1,      N1-j+1     ];
    R270 = @(i,j) [ j,           N1-i+1     ];
    RefX = @(i,j) [ N1-i+1,      j          ];
    RefY = @(i,j) [ i,           N1-j+1     ];
    Swap = @(i,j) [ j,           i          ];
    SR180= @(i,j) [ N1-j+1,      N1-i+1     ];  % anti-diagonal

    T.R0       = map(R0);
    T.R90      = map(R90);
    T.R180     = map(R180);
    T.R270     = map(R270);
    T.RefX     = map(RefX);
    T.RefY     = map(RefY);
    T.Swap     = map(Swap);
    T.SwapR180 = map(SR180);
end

function label = classify_d4(g2t_ref, g2t_e, T)
% Classify orientation of g2t_e relative to g2t_ref via D4 transforms.
% Each transform permutes tensor indices; apply it to the *values* of g2t_ref.

    candM = [T.R0, T.R90, T.R180, T.R270, T.RefX, T.RefY, T.Swap, T.SwapR180]; % N2 x 8

    % Compose: for each transform, permute the *values* of g2t_ref
    for k = 1:8
        cand = g2t_ref(candM(:,k));  % expected g2t for this transform
        if all(g2t_e(:) == cand(:))
            label = k; 
            return;
        end
    end

    % If no exact match (shouldn’t happen on these grids), pick closest
    diffs = sum(g2t_e(:) ~= g2t_ref(candM), 1);
    [~, label] = min(diffs);
end

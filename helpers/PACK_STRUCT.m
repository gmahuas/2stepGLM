function strct = PACK_STRUCT(varargin)
%PACK_STRUCT create a structure containing all the named variables
%
%     strct = PACK_STRUCT('var1', 'var2', ...)
%
% strct.var1, strct.var2, etc. will now contain the settings from variables in
% the calling workspace with the same name. A bit neater than:
%     strct = struct('var1', var1, 'var2', var2, ...);
%
% See also: UNPACK_STRUCT

% Iain Murray, January 2010

for ff = varargin(:)'
    var_name = ff{1};
    strct.(var_name) = evalin('caller', var_name);
end


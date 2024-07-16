const std = @import("std");
const interpreter = @import("interpreter.zig");

const AllocatorError = std.mem.Allocator.Error;

pub const Symbol = struct {
    name: []const u8,
    type: []const u8,
    val: ?interpreter.Interpreter.Result,
};

pub const SymbolTable = struct {
    symbolMap: std.StringHashMap(Symbol),
    parent: ?*const SymbolTable,
    allocator: std.mem.Allocator,

    pub fn init(parent: ?*const SymbolTable, allocator: std.mem.Allocator) SymbolTable {
        return .{
            .parent = parent,
            .symbolMap = std.StringHashMap(Symbol).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SymbolTable) void {
        self.symbolMap.deinit();
    }

    pub fn getSymbol(self: *const SymbolTable, name: []const u8) ?Symbol {
        const symbolMaybe = self.symbolMap.get(name);
        if (symbolMaybe != null) {
            return symbolMaybe.?;
        }

        if (self.parent != null) {
            return self.parent.?.*.getSymbol(name);
        }

        return null;
    }

    pub const SetSymbolError = error{ UnknownKeyError, AllocatorError, OutOfMemory };

    pub fn setSymbolVal(self: *SymbolTable, name: []const u8, val: ?interpreter.Interpreter.Result) SetSymbolError!void {
        const symMaybe = self.symbolMap.get(name);
        if (symMaybe == null) {
            return error.UnknownKeyError;
        }

        var sym = symMaybe.?;

        sym.val = val;

        try self.symbolMap.put(name, sym);
    }

    pub fn addSymbol(self: *SymbolTable, symbol: Symbol) void {
        self.symbolMap.put(symbol.name, symbol) catch |err| {
            std.debug.print("INTERNAL ERROR: Could not update symbol '{s}' in symboltable: {?}\n", .{ symbol.name, err });
        };
    }

    pub fn print(self: *SymbolTable) void {
        var symbolKeys = self.symbolMap.keyIterator();

        std.debug.print("Symbol table:\n", .{});
        while (symbolKeys.next()) |key| {
            const val = self.symbolMap.get(key.*).?;
            std.debug.print("    name: {s}, val: {s}\n", .{ key.*, val.val.?.toStr(self.allocator) });
        }
    }
};
